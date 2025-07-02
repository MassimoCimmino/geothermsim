# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Self, Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
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
    fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
        Determines the value of the fluid specific heat capacity to use in
        the energy balances:

            - 'nominal': The fluid specific heat capacity is evaluated at the
            nominal fluid temperature.
            - 'average': The fluid specific heat capacity is evaluated at the
            mean fluid temperature in the pipes. This only applies if the
            fluid temperature is provided as an input to the class methods.

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

    def __init__(self, R_d: ArrayLike | Callable[[float], Array], fluid: Fluid, r_b: float, path: Path, basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None, order: int = 101, order_to_self: int = 21, parallel: bool = True, fluid_heat_capacity_mode: str = 'nominal'):
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
        self.fluid_heat_capacity_mode = fluid_heat_capacity_mode
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

    def effective_borehole_thermal_resistance(self, m_flow: float, T_f: float | Array | None = None) -> float:
        """Effective borehole thermal resistance.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        float
            Effective borehole thermal resistance (in m-K/W).

        """
        cp_f = self.fluid_specific_heat(T_f=T_f)
        m_flow_pipe = self.m_flow_pipe(m_flow)
        beta_ij = self._beta_ij(m_flow, cp_f, T_f=T_f)
        a = self._outlet_fluid_temperature_a_in(
            beta_ij,
            self._top_connectivity,
            self._mixing,
            self._s_coefs)
        R_b = 0.5 * self.L / (m_flow * cp_f) * (1. + a) / (1. - a)
        return R_b

    def fluid_specific_heat(self, T_f: float | Array | None) -> float:
        """Fluid specific isobaric heat capacity.

        Parameters
        ----------
        T_f : float, array or None, default: None
            Fluid temperature or array of fluid temperatures
            (in degree Celsius). If ``None`` or if `fluid_heat_capacity_mode`
            is set to 'nominal', the nominal fluid temperature is used.

        Returns
        -------
        float
            The specific heat capacity at the mean temperature (in J/kg-K).

        """
        if T_f is None or self.fluid_heat_capacity_mode == 'nominal':
            cp_f = self.fluid.specific_heat()
        else:
            T_f = jnp.mean(T_f)
            cp_f = self.fluid.specific_heat(T_f=T_f)
        return cp_f

    def g(self, xi: Array | float, m_flow: float, T_f: float | Array | None = None) -> Tuple[Array | float, Array]:
        """Coefficients to evaluate the heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
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
        cp_f = self.fluid_specific_heat(T_f=T_f)
        a_in, a_b = self._heat_extraction_rate(xi, m_flow, cp_f, T_f=T_f)
        return a_in, a_b

    def g_to_self(self, m_flow: float, T_f: float | Array | None = None) -> Tuple[Array, Array]:
        """Coefficients to evaluate the heat extraction rate at nodes.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            (`n_nodes`,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_nodes`, `n_nodes`,) array of coefficients for the borehole
            wall temperature.

        """
        cp_f = self.fluid_specific_heat(T_f)
        a_in, a_b = self._heat_extraction_rate_to_self(m_flow, cp_f, T_f=T_f)
        return a_in, a_b

    def fluid_temperature(self, xi: Array | float, T_f_in: float, T_b: Array, m_flow: float, T_f: float | Array | None = None) -> Array:
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
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        array
            (M, 2,) array of fluid temperatures (in degree Celsius).

        """
        cp_f = self.fluid_specific_heat(T_f)
        a_in, a_b = self._fluid_temperature(xi, m_flow, cp_f, T_f=T_f)
        T_f = a_in * T_f_in + a_b @ T_b
        return T_f

    def heat_extraction_rate(self, xi: Array | float, T_f_in: float, T_b: Array, m_flow: float, T_f: float | Array | None = None) -> Array | float:
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
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        array
            (M,) array of heat extraction rate (in W/m). If `xi` is a
            ``float``, then ``M=0``.

        """
        cp_f = self.fluid_specific_heat(T_f)
        a_in, a_b = self._heat_extraction_rate(xi, m_flow, cp_f, T_f=T_f)
        q = a_in * T_f_in + a_b @ T_b
        return q

    def heat_extraction_rate_to_self(self, T_f_in: float, T_b: Array, m_flow: float, T_f: float | Array | None = None) -> Array:
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
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        array
            (`n_nodes`,) array of heat extraction rate (in W/m). If `xi`
            is a ``float``, then ``M=0``.

        """
        cp_f = self.fluid_specific_heat(T_f)
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

    def outlet_fluid_temperature(self, T_f_in: float, T_b: Array, m_flow: float, T_f: float | Array | None = None) -> float:
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
        T_f : float, array or None, default: None
            Fluid temperature or (`n_pipes`,) array of fluid
            temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        float
            Outlet fluid temperature (in degree Celsius).

        """
        cp_f = self.fluid_specific_heat(T_f)
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
    def from_dimensions(cls, R_d: ArrayLike | Callable[[float], Array], fluid: Fluid, L: float, D: float, r_b: float, x: float, y: float, basis: Basis, n_segments: int, tilt: float = 0., orientation: float = 0., segment_ratios: ArrayLike | None = None, order: int = 101, order_to_self: int = 21, parallel: bool = True, fluid_heat_capacity_mode: str = 'nominal') -> Self:
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
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use
            in the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at
                the nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at
                the mean fluid temperature in the pipes. This only applies if
                the fluid temperature is provided as an input to the class
                methods.

        Returns
        -------
        borehole
            Instance of the `Borehole` class.

        """
        path = Path.Line(L, D, x, y, tilt, orientation)
        return cls(R_d, fluid, r_b, path, basis, n_segments, segment_ratios=segment_ratios, order=order, order_to_self=order_to_self, parallel=parallel, fluid_heat_capacity_mode=fluid_heat_capacity_mode)
