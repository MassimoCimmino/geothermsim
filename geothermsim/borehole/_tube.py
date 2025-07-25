# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Tuple
from typing_extensions import Self

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import while_loop
from jax.typing import ArrayLike

from ..basis import Basis
from .borehole import Borehole
from ..fluid import Fluid
from ..path import Path


class _Tube(Borehole, ABC):
    """Geothermal borehole with tube.

    Parameters
    ----------
    R_d : array_like or callable
        (`n_pipes`, `n_pipes`,) array of thermal resistances (in m-K/W),
        or callable that takes the mass flow rate as input (in kg/s) and
        returns a (`n_pipes`, `n_pipes`,) array.
    fluid : fluid
        The fluid.
    r_b : float
        Borehole radius (in meters).
    path : path
        Path of the borehole.
    basis : basis
        Basis functions.
    n_segments : int
        Number of segments.
    segment_ratios : array_like or None, default: None
        Normalized size of the segments. Should total ``1``
        (i.e. ``sum(segment_ratios) = 1``). If `segment_ratios` is
        ``None``, segments of equal size are considered (i.e.
        ``segment_ratios[v] = 1 / n_segments``).
    order : int, default: 101
        Order of the Gauss-Legendre quadrature to evaluate thermal
        response factors to points outside the borehole, and to evaluate
        coefficient matrices for fluid and heat extraction rate profiles.
    order_to_self : int, default: 21
        Order of the tanh-sinh quadrature to evaluate thermal
        response factors to nodes on the borehole. Corresponds to the
        number of quadrature points along each subinterval delimited
        by nodes and edges of the segments.
    parallel : bool, default: True
        True if pipes are in parallel. False if pipes are in series.

    Attributes
    ----------
    n_pipes : int
        Number of pipes in the borehole.
    n_nodes : int
        Total number of nodes along the borehole.
    xi_edges : array
        (`n_segments`+1,) array of the coordinates of the edges of the
        segments.
    L : float
        Borehole length (in meters).
    xi : array
        (`n_nodes`,) array of node coordinates.
    p : array
        (`n_nodes`, 3,) array of node positions.
    dp_dxi : array
        (`n_nodes`, 3,) array of the derivatives of the position at the
        node coordinates.
    J : array
        (`n_nodes`,) array of the norm of the Jacobian at the node
        coordinates.
    s : array
        (`n_nodes`,) array of the longitudinal position at the node
        coordinates.
    w : array
        (`n_nodes`,) array of quadrature weights at the node coordinates.
        These quadrature weights take into account the norm of the
        Jacobian.

    """
 
    def __init__(
        self,
        R_d: ArrayLike | Callable[[float], Array],
        fluid: Fluid,
        r_b: float,
        path: Path,
        basis: Basis,
        n_segments: int,
        segment_ratios: ArrayLike | None = None,
        order: int = 101,
        order_to_self: int = 21,
        parallel: bool = True
    ):
        # Runtime type validation
        if not isinstance(R_d, ArrayLike) and not callable(R_d):
            raise TypeError(f"Expected arraylike or callable input; got {R_d}")
        # Convert input to jax.Array
        if isinstance(R_d, ArrayLike):
            R_d = jnp.atleast_2d(R_d)
            def thermal_resistances(m_flow: float, T_f: float | Array | None = None) -> Array:
                return R_d
            self.thermal_resistances = thermal_resistances
        else:
            self.thermal_resistances = R_d

        # Initialize borehole
        super().__init__(r_b, path, basis, n_segments, segment_ratios=segment_ratios)
        # Other attributes
        self.R_d = R_d
        self.fluid = fluid
        self.parallel = parallel
        shape = jnp.shape(self.thermal_resistances(1.))
        self.n_pipes = shape[0]
        self._n_pipes_over_two = int(self.n_pipes / 2)
        # Coefficients related to pipe configuration
        if self.parallel:
            # Factor for flow division amongst pipes
            self._m_flow_factor = 1 / self._n_pipes_over_two
            # Coefficients for connectivity at the top of the borehole
            # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
            _top_connectivity_in = jnp.ones(self._n_pipes_over_two)
            _top_connectivity_u = jnp.zeros(
                (self._n_pipes_over_two, self._n_pipes_over_two)
            )
            # Coefficient for mixing of the fluid at the outlet
            # T_f_out = m_u @ T_f_u(-1)
            self._mixing = jnp.full(
                self._n_pipes_over_two,
                self._m_flow_factor
            )
        else:
            # Factor for flow division amongst pipes
            self._m_flow_factor = 1
            # Coefficients for connectivity at the top of the borehole
            # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
            _top_connectivity_in = jnp.concatenate([
                jnp.ones(1),
                jnp.zeros(self._n_pipes_over_two - 1),
            ])
            _top_connectivity_u = jnp.eye(
                self._n_pipes_over_two,
                k=-1
            )
            # Coefficient for mixing of the fluid at the outlet
            # T_f_out = m_u @ T_f_u(-1)
            self._mixing = jnp.concatenate([
                jnp.zeros(self._n_pipes_over_two - 1),
                jnp.ones(1),
            ])
        self._top_connectivity = (_top_connectivity_in, _top_connectivity_u)

    def effective_borehole_thermal_resistance(
        self,
        T_f_in: float,
        T_b: Array,
        m_flow: float,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> float:
        """Effective borehole thermal resistance.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borehole.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole model:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average' and 'per-pipe' trigger iterations during the
            simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        float
            Effective borehole thermal resistance (in m-K/W).

        """
        T_f, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        R_b = self._effective_borehole_thermal_resistance(m_flow, cp_f, T_f)
        return R_b

    def fluid_temperature(
        self,
        xi: Array | float,
        T_f_in: float,
        T_b: Array,
        m_flow: float,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> Array:
        """Fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borehole.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole model:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average' and 'per-pipe' trigger iterations during the
            simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        array
            (M, 2,) array of fluid temperatures (in degree Celsius).

        """
        T_f, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        a_in, a_b = self._fluid_temperature(xi, m_flow, cp_f, T_f=T_f)
        T_f = a_in * T_f_in + a_b @ T_b
        return T_f

    def heat_extraction_rate(
        self,
        xi: Array | float,
        T_f_in: float,
        T_b: Array,
        m_flow: float,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> Array | float:
        """Heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borehole.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole model:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average' and 'per-pipe' trigger iterations during the
            simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        array
            (M,) array of heat extraction rate (in W/m). If `xi` is a
            ``float``, then ``M=0``.

        """
        T_f, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        a_in, a_b = self._heat_extraction_rate(xi, m_flow, cp_f, T_f=T_f)
        q = a_in * T_f_in + a_b @ T_b
        return q

    def heat_extraction_rate_to_self(
        self,
        T_f_in: float,
        T_b: Array,
        m_flow: float,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> Array:
        """Heat extraction rate at nodes.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borehole.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole model:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average' and 'per-pipe' trigger iterations during the
            simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        array
            (`n_nodes`,) array of heat extraction rate (in W/m). If `xi`
            is a ``float``, then ``M=0``.

        """
        T_f, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        a_in, a_b = self._heat_extraction_rate_to_self(m_flow, cp_f, T_f=T_f)
        q = a_in * T_f_in + a_b @ T_b
        return q

    def m_flow_pipe(self, m_flow: float) -> Array:
        """Fluid mass flow rate per pipe.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate into the borehole (in kg/s).

        Returns
        -------
        float
            Fluid mass flow rate in each pipes (in kg/s).

        """
        return self._m_flow_factor * m_flow

    def outlet_fluid_temperature(
        self,
        T_f_in: float,
        T_b: Array,
        m_flow: float,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> float:
        """Outlet fluid temperature.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borehole.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole model:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average' and 'per-pipe' trigger iterations during the
            simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        float
            Outlet fluid temperature (in degree Celsius).

        """
        T_f, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        a_in, a_b = self._outlet_fluid_temperature(m_flow, cp_f, T_f=T_f)
        T_f_out = a_in * T_f_in + a_b @ T_b
        return T_f_out

    def _beta_ij(self, m_flow: float, cp_f: float, T_f: float | Array | None = None) -> Array:
        """Thermal conductance coefficients.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        """
        R_d = self.thermal_resistances(m_flow, T_f=T_f)
        return 1. / (self.m_flow_pipe(m_flow) * cp_f * R_d)

    def _effective_borehole_thermal_resistance(self, m_flow: float, cp_f: float, T_f: float | Array | None = None) -> float:
        """Effective borehole thermal resistance.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        float
            Effective borehole thermal resistance (in m-K/W).

        """
        m_flow_pipe = self.m_flow_pipe(m_flow)
        beta_ij = self._beta_ij(m_flow, cp_f, T_f=T_f)
        a = self._outlet_fluid_temperature_a_in(
            beta_ij,
            self._top_connectivity,
            self._mixing,
            self._s_coefs)
        R_b = 0.5 * self.L / (m_flow * cp_f) * (1. + a) / (1. - a)
        return R_b

    def _fluid_temperature(self, xi: Array | float, m_flow: float, cp_f: float, T_f: float | Array | None = None) -> Tuple[Array, Array]:
        """Coefficients to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            (M, `n_pipes`,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (M, `n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        beta_ij = self._beta_ij(m_flow, cp_f, T_f=T_f)
        if len(jnp.shape(xi)) > 0:
            xi_p, index = vmap(
                self._segment_coordinate,
                in_axes=(0, None),
                out_axes=0
            )(xi, self.xi_edges)
            a_in = vmap(
                self._fluid_temperature_a_in,
                in_axes=(0, 0, None, None, None),
                out_axes=0
            )(xi_p, index, beta_ij, self._top_connectivity, self._s_coefs)
            a_b = vmap(
                self._fluid_temperature_a_b,
                in_axes=(0, 0, None, None, None, None, None, None),
                out_axes=0
            )(
                xi_p,
                index,
                beta_ij,
                self._top_connectivity,
                self._s_coefs,
                self._J_coefs,
                self.basis._psi_coefs,
                self.order
            )
        else:
            xi_p, index = self._segment_coordinate(xi, self.xi_edges)
            a_in = self._fluid_temperature_a_in(
                xi_p,
                index,
                beta_ij,
                self._top_connectivity,
                self._s_coefs)
            a_b = self._fluid_temperature_a_b(
                xi_p,
                index,
                beta_ij,
                self._top_connectivity,
                self._s_coefs,
                self._J_coefs,
                self.basis._psi_coefs,
                self.order
            )
        return a_in, a_b

    def _heat_extraction_rate(self, xi: Array | float, m_flow: float, cp_f: float, T_f: float | Array | None = None) -> Tuple[Array | float, Array]:
        """Coefficients to evaluate the heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array or float
            (M,) array of coefficients for the inlet fluid temperature.
        a_b : array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        m_flow_pipe = self.m_flow_pipe(m_flow)
        beta_ij = self._beta_ij(m_flow, cp_f, T_f=T_f)
        if len(jnp.shape(xi)) > 0:
            xi_p, index = vmap(
                self._segment_coordinate,
                in_axes=(0, None),
                out_axes=0
            )(xi, self.xi_edges)
            a_in = vmap(
                self._heat_extraction_rate_a_in,
                in_axes=(0, 0, None, None, None, None, None),
                out_axes=0
            )(
                xi_p,
                index,
                m_flow_pipe,
                cp_f,
                beta_ij,
                self._top_connectivity,
                self._s_coefs
            )
            a_b = vmap(
                self._heat_extraction_rate_a_b,
                in_axes=(0, 0, None, None, None, None, None, None, None, None),
                out_axes=0
            )(
                xi_p,
                index,
                m_flow_pipe,
                cp_f,
                beta_ij,
                self._top_connectivity,
                self._s_coefs,
                self._J_coefs,
                self.basis._psi_coefs,
                self.order
            )
        else:
            xi_p, index = self._segment_coordinate(xi, self.xi_edges)
            a_in = self._heat_extraction_rate_a_in(
                xi_p,
                index,
                m_flow_pipe,
                cp_f,
                beta_ij,
                self._top_connectivity,
                self._s_coefs
            )
            a_b = self._heat_extraction_rate_a_b(
                xi_p,
                index,
                m_flow_pipe,
                cp_f,
                beta_ij,
                self._top_connectivity,
                self._s_coefs,
                self._J_coefs,
                self.basis._psi_coefs,
                self.order
            )
        return a_in, a_b

    def _heat_extraction_rate_to_self(self, m_flow: float, cp_f: float, T_f: float | Array | None = None) -> Tuple[Array | float, Array]:
        """Coefficients to evaluate the heat extraction rate.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array or float
            (`n_nodes`,) array of coefficients for the inlet fluid temperature.
        a_b : array
            (`n_nodes`, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        m_flow_pipe = self.m_flow_pipe(m_flow)
        beta_ij = self._beta_ij(m_flow, cp_f, T_f=T_f)
        xi_p = self.basis.xi
        index = jnp.arange(self.n_segments)
        a_in = vmap(
            vmap(
                self._heat_extraction_rate_a_in,
                in_axes=(0, None, None, None, None, None, None),
                out_axes=0),
            in_axes=(None, 0, None, None, None, None, None),
            out_axes=0
        )(
            xi_p,
            index,
            m_flow_pipe,
            cp_f,
            beta_ij,
            self._top_connectivity,
            self._s_coefs
        )
        a_b = vmap(
            vmap(
                self._heat_extraction_rate_a_b,
                in_axes=(0, None, None, None, None, None, None, None, None, None),
                out_axes=0),
            in_axes=(None, 0, None, None, None, None, None, None, None, None),
            out_axes=0
        )(
            xi_p,
            index,
            m_flow_pipe,
            cp_f,
            beta_ij,
            self._top_connectivity,
            self._s_coefs,
            self._J_coefs,
            self.basis._psi_coefs,
            self.order
        )
        return a_in.flatten(), a_b.reshape(self.n_nodes, self.n_nodes)

    def _outlet_fluid_temperature(self, m_flow: float, cp_f: float, T_f: float | Array | None = None) -> Tuple[float, Array]:
        """Coefficients to evaluate the outlet fluid temperature.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : float
            Coefficient for the inlet fluid temperature.
        a_b : array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        beta_ij = self._beta_ij(m_flow, cp_f, T_f=T_f)
        a_in = self._outlet_fluid_temperature_a_in(
            beta_ij,
            self._top_connectivity,
            self._mixing,
            self._s_coefs
        )
        a_b = self._outlet_fluid_temperature_a_b(
            beta_ij,
            self._top_connectivity,
            self._mixing,
            self._s_coefs,
            self._J_coefs,
            self.basis._psi_coefs,
            self.order
        )
        return a_in, a_b

    @partial(jit, static_argnames=['self', 'fluid_heat_capacity_mode', 'thermal_resistances_mode'])
    def _reference_fluid_conditions(
        self,
        T_f_in: float,
        T_b: Array,
        m_flow: float,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> Tuple[float | Array, float]:
        """Reference conditions for fluid properties.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_nodes`,) array of borehole wall temperatures (in degree
            Celsius).
        m_flow : float
            Fluid mass flow rate (in kg/s).
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the pipes. This only applies if the
                fluid temperature is provided as an input to the class methods.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole model:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average' and 'per-pipe' trigger iterations during the
            simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        T_f : float or array
            The reference fluid temperature (in degree Celsius), or
            (n_pipes,) array or reference fluid temperatures to evaluate
            the convection thermal resistances in pipes.
        cp_f : float
            The specific heat capacity (in J/kg-K).

        """
        # The initial guess on the reference fluid temperature is equal
        # to the inlet fluid temperature, unless the mode is 'nominal'
        if thermal_resistances_mode == 'nominal':
            T_f_init = T_f = self.fluid.T_nominal
        elif thermal_resistances_mode in ['average', 'per-borehole']:
            T_f_init = T_f_in
        else:
            T_f_init = jnp.broadcast_to(T_f_in, self.n_pipes)

        # The initial guess on the fluid temperature for the specific
        # heat is the inlet fluid temperature, unless the mode is 'nominal'
        if fluid_heat_capacity_mode == 'nominal':
            cp_f = self.fluid.specific_heat(T=self.fluid.T_nominal)
            T_cp_f_init = self.fluid.T_nominal
        else:
            T_cp_f_init = T_f_in

        # Early return if both modes are 'nominal'
        if (
            thermal_resistances_mode == 'nominal'
            and fluid_heat_capacity_mode == 'nominal'
        ):
            return T_f, cp_f

        def fluid_specific_heat(_T_cp_f: float) -> float:
            """Fluid specific heat capacity."""
            if fluid_heat_capacity_mode == 'nominal':
                _cp_f = cp_f
            else:
                _cp_f = self.fluid.specific_heat(T=_T_cp_f)
            return _cp_f

        def fluid_specific_heat_temperature(_cp_f: float, _T_f: float | Array) -> float:
            """Fluid temperature for the fluid specific heat."""
            if fluid_heat_capacity_mode == 'nominal':
                # Return the nominal fluid temperature
                _T_cp_f = self.fluid.T_nominal
            elif thermal_resistances_mode in ['average', 'per-borehole']:
                # The temperature is the same as for the thermal resistances
                _T_cp_f = _T_f
            else:
                # Evaluate the mean fluid temperature in the borehole
                a_in, a_b = self._outlet_fluid_temperature(m_flow, _cp_f, T_f=_T_f)
                _T_f_out = a_in * T_f_in + a_b @ T_b
                _T_cp_f = 0.5 * (T_f_in + _T_f_out)
            return _T_cp_f

        def thermal_resistance_temperature(_cp_f: float, _T_f: float | Array) -> float | Array:
            """Reference fluid temperature."""
            if thermal_resistances_mode == 'nominal':
                # Return the nominal fluid temperature
                _T_f = self.fluid.T_nominal
            elif thermal_resistances_mode in ['average', 'per-borehole']:
                # Evaluate the mean fluid temperature in the borehole
                a_in, a_b = self._outlet_fluid_temperature(m_flow, _cp_f, T_f=_T_f)
                _T_f_out = a_in * T_f_in + a_b @ T_b
                _T_f = 0.5 * (T_f_in + _T_f_out)
            else:
                # Evaluate the fluid temperature at mid-depth in each pipe
                a_in, a_b = self._fluid_temperature(0.5, m_flow, _cp_f, T_f=_T_f)
                _T_f = a_in * T_f_in + a_b @ T_b
            return _T_f

        def cond_fun(val: Tuple[float | Array, float, float]) -> bool:
            """True if the temperature change is above tolerance."""
            _T_f, _T_cp_f, _delta = val
            return _delta > tolT

        def body_fun(val: Tuple[float | Array, float, float]) -> Tuple[float | Array, float, float]:
            """Update the temperatures for specific heat and resistances."""
            # Unpack values
            _T_f, _T_cp_f, _delta = val

            # Update temperatures
            _cp_f = fluid_specific_heat(_T_cp_f)
            _T_f_new = thermal_resistance_temperature(_cp_f, _T_f)
            _T_cp_f_new = fluid_specific_heat_temperature(_cp_f, _T_f_new)

            # Evaluate the absolute change
            _delta = jnp.maximum(
                jnp.abs(_T_cp_f_new - _T_cp_f),
                jnp.max(jnp.abs(_T_f_new - _T_f))
            )
            return _T_f_new, _T_cp_f_new, _delta

        # Iterate until convergence
        init_val = T_f_init, T_cp_f_init, jnp.inf
        T_f, T_cp_f, delta = while_loop(cond_fun, body_fun, init_val)
        cp_f = fluid_specific_heat(T_cp_f)

        return T_f, cp_f

    @classmethod
    @abstractmethod
    def _fluid_temperature_a_in(cls, xi_p: float, index: int, beta_ij: Array, top_connectivity: Tuple[Array, Array], s_coefs: Array) -> Array:
        """Inlet coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij : array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole as a function of `xi_p`.

        Returns
        -------
        array
            (`n_pipes`,) array of coefficients for the inlet fluid
            temperature.

        """
        ...

    @classmethod
    @abstractmethod
    def _fluid_temperature_a_b(cls, xi_p: float, index: int, beta_ij: Array, top_connectivity: Tuple[Array, Array], s_coefs: Array, J_coefs: Array, psi_coefs: Array, order: int = 101) -> Array:
        """Borehole wall coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij : array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.
        J_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the norm of the jacobian (in meters) along
            the borehole segments as a function of `xi_p`.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions along
            the borehole segments as a function of `xi_p`.
        order : int, default: 101
            Order of the numerical quadrature.

        Returns
        -------
        array
            (`n_pipes`, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        ...

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _heat_extraction_rate_a_in(cls, xi_p: float, index: int, m_flow_pipe: float, cp_f: float, beta_ij: Array, top_connectivity: Tuple[Array, Array], s_coefs: Array) -> float:
        """Inlet coefficient to evaluate the heat extraction rate.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        m_flow_pipe : float
            Fluid mass flow rate in the pipes (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        float
            Coefficient for the inlet fluid temperature.

        """
        b_in = cls._fluid_temperature_a_in(
            xi_p,
            index,
            beta_ij,
            top_connectivity,
            s_coefs
        )
        R_d = 1 / (m_flow_pipe * cp_f * beta_ij)
        R_d_diag = jnp.diag(R_d)
        G_d_diag = 1 / R_d_diag
        a_in = -G_d_diag @ b_in
        return a_in

    @classmethod
    @partial(jit, static_argnames=['cls', 'order'])
    def _heat_extraction_rate_a_b(cls, xi_p: float, index: int, m_flow_pipe: float, cp_f: float, beta_ij: Array, top_connectivity: Tuple[Array, Array], s_coefs: Array, J_coefs: Array, psi_coefs: Array, order: int = 101) -> Array:
        """Borehole wall coefficient to evaluate the heat extraction rate.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        m_flow_pipe : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.
        J_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the norm of the jacobian (in meters) along
            the borehole segments as a function of `xi_p`.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions along
            the borehole segments as a function of `xi_p`.
        order : int, default: 101
            Order of the numerical quadrature.

        Returns
        -------
        array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        a_b = jnp.zeros((jnp.shape(s_coefs)[1], jnp.shape(psi_coefs)[0]))
        b_b = cls._fluid_temperature_a_b(
            xi_p,
            index,
            beta_ij,
            top_connectivity,
            s_coefs,
            J_coefs,
            psi_coefs,
            order
        )
        R_d = 1 / (m_flow_pipe * cp_f * beta_ij)
        R_d_diag = jnp.diag(R_d)
        G_d_diag = 1 / R_d_diag
        R_b = 1 / (1 / R_d_diag).sum()
        psi = Basis._f_psi(xi_p, psi_coefs)
        a_b = a_b.at[index, :].set(psi * G_d_diag.sum()).flatten()
        a_b = a_b.at[:].add(-G_d_diag @ b_b)
        return a_b

    @classmethod
    @abstractmethod
    def _outlet_fluid_temperature_a_in(cls, beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        mixing : array
            (`n_pipes`/2,) array of coefficients to evaluate the outlet fluid
            temperature from the fluid temperatures at the top-end of the
            borehole in the upward flowing pipes following the relation:
            ``T_f_out = mixing @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        float
            Coefficient for the inlet fluid temperature.

        """
        ...

    @classmethod
    @abstractmethod
    def _outlet_fluid_temperature_a_b(cls, beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array, J_coefs: Array, psi_coefs: Array, order: int = 101) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        mixing : array
            (`n_pipes`/2,) array of coefficients to evaluate the outlet fluid
            temperature from the fluid temperatures at the top-end of the
            borehole in the upward flowing pipes following the relation:
            ``T_f_out = mixing @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.
        J_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the norm of the jacobian (in meters) along
            the borehole segments as a function of `xi_p`.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions along
            the borehole segments as a function of `xi_p`.
        order : int, default: 101
            Order of the numerical quadrature.

        Returns
        -------
        array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        ...

    @classmethod
    def from_dimensions(
        cls,
        R_d: ArrayLike | Callable[[float], Array],
        fluid: Fluid,
        L: float,
        D: float,
        r_b: float,
        x: float,
        y: float,
        basis: Basis,
        n_segments: int,
        tilt: float = 0.,
        orientation: float = 0.,
        segment_ratios: ArrayLike | None = None,
        order: int = 101,
        order_to_self: int = 21,
        parallel: bool = True
    ) -> Self:
        """Straight borehole from its dimensions.

        Parameters
        ----------
        R_d : array_like or callable
            (`n_pipes`, `n_pipes`,) array of thermal resistances
            (in m-K/W), or callable that takes the mass flow rate as input
            (in kg/s) and returns a (`n_pipes`, `n_pipes`,) array.
        fluid : fluid
            The fluid.
        L : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        x, y : float
            Horizontal position (in meters) of the top end of the
            borehole.
        basis : basis
            Basis functions.
        n_segments : int
            Number of segments.
        tilt : float, default: ``0.``
            Tilt angle (in radians) of the borehole with respect to
            vertical.
        orientation : float, default: ``0.``
            Orientation (in radians) of the inclined borehole. An
            inclination toward the x-axis corresponds to an orientation
            of zero.
        segment_ratios : array_like or None, default: None
            Normalized size of the segments. Should total ``1``
            (i.e. ``sum(segment_ratios) = 1``). If `segment_ratios` is
            ``None``, segments of equal size are considered (i.e.
            ``segment_ratios[v] = 1 / n_segments``).
        order : int, default: 101
            Order of the Gauss-Legendre quadrature to evaluate thermal
            response factors to points outside the borehole, and to evaluate
            coefficient matrices for fluid and heat extraction rate profiles.
        order_to_self : int, default: 21
            Order of the tanh-sinh quadrature to evaluate thermal
            response factors to nodes on the borehole. Corresponds to the
            number of quadrature points along each subinterval delimited
            by nodes and edges of the segments.
        parallel : bool, default: True
            True if pipes are in parallel. False if pipes are in series.

        Returns
        -------
        borehole
            Instance of the `Borehole` class.

        """
        path = Path.Line(L, D, x, y, tilt, orientation)
        return cls(R_d, fluid, r_b, path, basis, n_segments, segment_ratios=segment_ratios, order=order, order_to_self=order_to_self, parallel=parallel)
