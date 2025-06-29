# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import partial
from typing import List, Tuple
from typing_extensions import Self

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import while_loop
from jax.typing import ArrayLike

from ..basis import Basis
from .borefield import Borefield
from ..borehole import SingleUTube
from ..fluid import Fluid
from ..path import Path


class Network(Borefield):
    """Borefield with pipes.

    Parameters
    ----------
    boreholes : list of single_u_tube
        Boreholes in the borefield.
    fluid : fluid
        The fluid.

    Attributes
    ----------
    n_boreholes : int
        Number of boreholes.
    n_nodes : int
        Number of nodes per borehole.
    L : array
        Borehole lengths (in meters).
    xi : array
        Coordinates of the nodes. They are the same for all boreholes.
    p : array
        (`n_boreholes`, `n_nodes`, 3,) array of node positions.
    dp_dxi : array
        (`n_boreholes`, `n_nodes`, 3,) array of the derivatives of the
        position at the node coordinates.
    J : array
        (`n_boreholes`, `n_nodes`,) array of the norm of the Jacobian at
        the node coordinates.
    s : array
        (`n_boreholes`, `n_nodes`,) array of the longitudinal position at
        the node coordinates.
    w : array
        (`n_boreholes`, `n_nodes`,) array of quadrature weights at the
        node coordinates. These quadrature weights take into account the
        norm of the Jacobian.

    """

    def __init__(self, boreholes: List[SingleUTube], fluid: Fluid):
        super().__init__(boreholes)
        self.fluid = fluid

    def effective_borefield_thermal_resistance(
        self,
        T_f_in: float,
        T_b: Array,
        m_flow: float | Array,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> float:
        """Effective borefield thermal resistance.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperatures (in degree Celsius).
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borefield.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole models:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borefield.
                - 'per-borehole': The thermal resistances are evaluated at the mean
                fluid temperature in each borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average', 'per-borehole' and 'per-pipe' trigger iterations
            during the simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        float
            Effective borefield thermal resistance (in m-K/W).

        """
        T_reference, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        # Effective borehole thermal resistance
        R_field = self._effective_borefield_thermal_resistance(m_flow, cp_f, T_f=T_reference)
        return R_field

    def fluid_temperature(
        self,
        xi: Array | float,
        T_f_in: float,
        T_b: Array,
        m_flow: float | Array,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> Array:
        """Fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the boreholes.
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperatures (in degree Celsius).
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borefield.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole models:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borefield.
                - 'per-borehole': The thermal resistances are evaluated at the mean
                fluid temperature in each borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average', 'per-borehole' and 'per-pipe' trigger iterations
            during the simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        array
            (`n_boreholes`, M, 2,) array of fluid temperatures (in degree
            Celsius).

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        T_reference, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        a_in, a_b = self._fluid_temperature(xi, m_flow, cp_f, T_f=T_reference)
        T_f = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return T_f

    def heat_extraction_rate(
        self,
        xi: Array | float,
        T_f_in: float,
        T_b: Array,
        m_flow: float | Array,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> Array:
        """Heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the boreholes.
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperatures (in degree Celsius).
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borefield.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole models:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borefield.
                - 'per-borehole': The thermal resistances are evaluated at the mean
                fluid temperature in each borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average', 'per-borehole' and 'per-pipe' trigger iterations
            during the simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        array
            (`n_boreholes`, M,) array of heat extraction rate (in W/m). If
            `xi` is a ``float``, then ``M=0``.

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        T_reference, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        a_in, a_b = self._heat_extraction_rate(xi, m_flow, cp_f, T_f=T_reference)
        q = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return q

    def heat_extraction_rate_to_self(
        self,
        T_f_in: float,
        T_b: Array,
        m_flow: float | Array,
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
            (`n_boreholes`, `n_nodes`,) array of borehole wall
            temperatures (in degree Celsius).
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borefield.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole models:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borefield.
                - 'per-borehole': The thermal resistances are evaluated at the mean
                fluid temperature in each borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average', 'per-borehole' and 'per-pipe' trigger iterations
            during the simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        array
            (`n_boreholes`, `n_nodes`,) array of heat extraction rate
            (in W/m). If `xi` is a ``float``, then ``M=0``.

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        T_reference, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        a_in, a_b = self._heat_extraction_rate_to_self(m_flow, cp_f, T_f=T_reference)
        q = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return q

    def m_flow_borehole(self, m_flow: float | Array) -> Array:
        """Fluid mass flow rate into the boreholes.

        Parameters
        ----------
        m_flow : float or array
            Fluid mass flow rate entering the borefield, or (`n_boreholes`,)
            array of fluid mass flow rate entering each borehole (in kg/s).

        Returns
        -------
        array
            (`n_boreholes`,) array of fluid mass flow rate entering each
            borehole (in kg/s).

        """
        if len(jnp.shape(m_flow)) == 0 or jnp.shape(m_flow)[0] == 1:
            m_flow = jnp.broadcast_to(
                m_flow / self.n_boreholes,
                self.n_boreholes
            )
        return m_flow

    def outlet_fluid_temperature(
        self,
        T_f_in: float,
        T_b: Array,
        m_flow: float | Array,
        fluid_heat_capacity_mode: str = 'nominal',
        thermal_resistances_mode: str = 'nominal',
        tolT: float = 1e-3
        ) -> Array:
        """Outlet fluid temperatures.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_boreholes, `n_nodes`,) array of borehole wall temperatures
            (in degree Celsius).
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borefield.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole models:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borefield.
                - 'per-borehole': The thermal resistances are evaluated at the mean
                fluid temperature in each borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average', 'per-borehole' and 'per-pipe' trigger iterations
            during the simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        array
            (`n_boreholes`,) array of outlet fluid temperatures (in degree
            Celsius).

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        T_reference, cp_f = self._reference_fluid_conditions(T_f_in, T_b, m_flow, fluid_heat_capacity_mode, thermal_resistances_mode, tolT)
        a_in, a_b = self._outlet_fluid_temperature(m_flow, cp_f, T_f=T_reference)
        T_f_out = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return T_f_out

    def _effective_borefield_thermal_resistance(self, m_flow: float | Array, cp_f: float, T_f: float | Array | None = None) -> float:
        """Effective borefield thermal resistance.

        Parameters
        ----------
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Mean fluid temperature, (`n_boreholes`,) array of per-borehole
            mean fluid temperature, or (`n_boreholes`, `n_pipes`,) array of
            fluid temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        float
            Effective borefield thermal resistance (in m-K/W).

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        m_flow_network = jnp.sum(m_flow)
        a = jnp.average(
            self._outlet_fluid_temperature(m_flow_borehole, cp_f, T_f=T_f)[0],
            weights=m_flow_borehole
        )
        # Effective borehole thermal resistance
        R_field = 0.5 * self.L.sum() / (m_flow_network * cp_f) * (1. + a) / (1. - a)
        return R_field

    def _fluid_temperature(self, xi: Array | float, m_flow: float | Array, cp_f: float, T_f: float | Array | None = None) -> Tuple[Array, Array]:
        """Coefficients to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the boreholes.
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Mean fluid temperature, (`n_boreholes`,) array of per-borehole
            mean fluid temperature, or (`n_boreholes`, `n_pipes`,) array of
            fluid temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            (`n_boreholes`, M, 2,) array of coefficients for the inlet
            fluid temperature.
        a_b : array
            (`n_boreholes`, M, 2, `n_nodes`,) array of coefficients for
            the borehole wall temperature.

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        if T_f is None or len(jnp.shape(T_f)) == 0:
            a_in, a_b = zip(*[
                borehole._fluid_temperature(xi, _m_flow, cp_f, T_f=T_f)
                for borehole, _m_flow in zip(self.boreholes, m_flow_borehole)
            ])
        else:
            a_in, a_b = zip(*[
                borehole._fluid_temperature(xi, _m_flow, cp_f, T_f=_T_f)
                for borehole, _m_flow, _T_f in zip(self.boreholes, m_flow_borehole, T_f)
            ])
        a_in = jnp.stack(a_in, axis=0)
        a_b = jnp.stack(a_b, axis=0)
        return a_in, a_b

    def _heat_extraction_rate(self, xi: Array | float, m_flow: float | Array, cp_f: float, T_f: float | Array | None = None) -> Tuple[Array, Array]:
        """Coefficients to evaluate the heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the boreholes.
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Mean fluid temperature, (`n_boreholes`,) array of per-borehole
            mean fluid temperature, or (`n_boreholes`, `n_pipes`,) array of
            fluid temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array or float
            (`n_boreholes`, M,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_boreholes`, M, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        if T_f is None or len(jnp.shape(T_f)) == 0:
            a_in, a_b = zip(*[
                borehole._heat_extraction_rate(xi, _m_flow, cp_f, T_f=T_f)
                for borehole, _m_flow in zip(self.boreholes, m_flow_borehole)
            ])
        else:
            a_in, a_b = zip(*[
                borehole._heat_extraction_rate(xi, _m_flow, cp_f, T_f=_T_f)
                for borehole, _m_flow, _T_f in zip(self.boreholes, m_flow_borehole, T_f)
            ])
        a_in = jnp.stack(a_in, axis=0)
        a_b = jnp.stack(a_b, axis=0)
        return a_in, a_b

    def _heat_extraction_rate_to_self(self, m_flow: float | Array, cp_f: float, T_f: float | Array | None = None) -> Tuple[Array, Array]:
        """Coefficients to evaluate the heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the boreholes.
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Mean fluid temperature, (`n_boreholes`,) array of per-borehole
            mean fluid temperature, or (`n_boreholes`, `n_pipes`,) array of
            fluid temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array or float
            (`n_boreholes`, M,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_boreholes`, M, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        if T_f is None or len(jnp.shape(T_f)) == 0:
            a_in, a_b = zip(*[
                borehole._heat_extraction_rate_to_self(_m_flow, cp_f, T_f=T_f)
                for borehole, _m_flow in zip(self.boreholes, m_flow_borehole)
            ])
        else:
            a_in, a_b = zip(*[
                borehole._heat_extraction_rate_to_self(_m_flow, cp_f, T_f=_T_f)
                for borehole, _m_flow, _T_f in zip(self.boreholes, m_flow_borehole, T_f)
            ])
        a_in = jnp.stack(a_in, axis=0)
        a_b = jnp.stack(a_b, axis=0)
        return a_in, a_b

    def _outlet_fluid_temperature(self, m_flow: float | Array, cp_f: float, T_f: float | Array | None = None) -> Tuple[Array, Array]:
        """Coefficients to evaluate the outlet fluid temperatures.

        Parameters
        ----------
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        T_f : float, array or None, default: None
            Mean fluid temperature, (`n_boreholes`,) array of per-borehole
            mean fluid temperature, or (`n_boreholes`, `n_pipes`,) array of
            fluid temperatures in each pipe (in degree Celcius). If
            ``None``, the nominal fluid temperature is used.

        Returns
        -------
        a_in : array
            (`n_boreholes`,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_boreholes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)
        if T_f is None or len(jnp.shape(T_f)) == 0:
            a_in, a_b = zip(*[
                borehole._outlet_fluid_temperature(_m_flow, cp_f, T_f=T_f)
                for borehole, _m_flow in zip(self.boreholes, m_flow_borehole)
            ])
        else:
            a_in, a_b = zip(*[
                borehole._outlet_fluid_temperature(_m_flow, cp_f, T_f=_T_f)
                for borehole, _m_flow, _T_f in zip(self.boreholes, m_flow_borehole, T_f)
            ])
        a_in = jnp.stack(a_in, axis=0)
        a_b = jnp.stack(a_b, axis=0)
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
        ) -> float:
        """Reference conditions for fluid properties.

        Parameters
        ----------
        T_f_in : float
            Inlet fluid temperature (in degree Celsius).
        T_b : array
            (`n_boreholes, `n_nodes`,) array of borehole wall temperatures
            (in degree Celsius).
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        fluid_heat_capacity_mode : {'nominal', 'average'}, default: 'nominal'
            Determines the value of the fluid specific heat capacity to use in
            the energy balances:
    
                - 'nominal': The fluid specific heat capacity is evaluated at the
                nominal fluid temperature.
                - 'average': The fluid specific heat capacity is evaluated at the
                mean fluid temperature in the borefield.
    
            The mode 'average' triggers iterations during the simulation.
        thermal_resistances_mode : {'nominal', 'average', 'per-pipe'}, default: 'nominal'
            Determines the value of the thermal resistances to use in the
            borehole models:
    
                - 'nominal': The thermal resistances are evaluated at the nominal
                fluid temperature.
                - 'average': The thermal resistances are evaluated at the mean
                fluid temperature in the borefield.
                - 'per-borehole': The thermal resistances are evaluated at the mean
                fluid temperature in each borehole.
                - 'per-pipe': The thermal resistances are evaluated at the fluid
                temperature in each pipe, evaluated at mid-depth.
    
            The modes 'average', 'per-borehole' and 'per-pipe' trigger iterations
            during the simulation.
        tolT : float, default: 1e-3
            Absolute tolerance on fluid temperatures during iterative
            simulation steps.

        Returns
        -------
        T_f : float or array
            The reference fluid temperature (in degree Celsius), or
            (n_boreholes,) or (n_boreholes, n_pipes,) array or reference
            fluid temperatures to evaluate the convection thermal resistances
            in pipes.
        cp_f : float
            The specific heat capacity (in J/kg-K).

        """
        m_flow_borehole = self.m_flow_borehole(m_flow)

        # The initial guess on the reference fluid temperature is equal
        # to the inlet fluid temperature, unless the mode is 'nominal'
        if fluid_heat_capacity_mode == 'nominal':
            cp_f = self.fluid.specific_heat(T=self.fluid.T_nominal)
            T_cp_f_init = self.fluid.T_nominal
        else:
            cp_f = self.fluid.specific_heat(T=T_f_in)
            T_cp_f_init = T_f_in

        # The initial guess on the fluid temperature for the specific
        # heat is the inlet fluid temperature, unless the mode is 'nominal'
        if thermal_resistances_mode == 'nominal':
            T_f_init = T_f = self.fluid.T_nominal
        elif thermal_resistances_mode == 'average':
            T_f_init = T_f_in
        elif thermal_resistances_mode == 'per-borehole':
            T_f_init = jnp.broadcast_to(T_f_in, self.n_boreholes)
        else:
            a_in, a_b = self._fluid_temperature(0.5, m_flow, cp_f, T_f=T_f_in)
            T_f_init = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)

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
            elif thermal_resistances_mode == 'average':
                # The temperature is the same as for the thermal resistances
                _T_cp_f = _T_f
            elif thermal_resistances_mode == 'per-borehole':
                # The temperature is obtained by mixing of the outlets
                _T_cp_f = jnp.average(_T_f, weights=m_flow_borehole)
            else:
                # Evaluate the mean fluid temperature in the borefield
                a_in, a_b = self._outlet_fluid_temperature(m_flow, _cp_f, T_f=_T_f)
                _T_f_out = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
                _T_f_out = jnp.average(_T_f_out, weights=m_flow_borehole)
                _T_cp_f = 0.5 * (T_f_in + _T_f_out)
            return _T_cp_f

        def thermal_resistance_temperature(_cp_f: float, _T_f: float | Array) -> float | Array:
            """Reference fluid temperature."""
            if thermal_resistances_mode == 'nominal':
                # Return the nominal fluid temperature
                _T_f = self.fluid.T_nominal
            elif thermal_resistances_mode == 'average':
                # Evaluate the mean fluid temperature in the borefield
                a_in, a_b = self._outlet_fluid_temperature(m_flow, _cp_f, T_f=_T_f)
                _T_f_out = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
                _T_f_out = jnp.average(_T_f_out, weights=m_flow_borehole)
                _T_f = 0.5 * (T_f_in + _T_f_out)
            elif thermal_resistances_mode == 'per-borehole':
                # Evaluate the mean fluid temperature in each borehole
                a_in, a_b = self._outlet_fluid_temperature(m_flow, _cp_f, T_f=_T_f)
                _T_f_out = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
                _T_f = 0.5 * (T_f_in + _T_f_out)
            else:
                # Evaluate the fluid temperature at mid-depth in each pipe
                a_in, a_b = self._fluid_temperature(0.5, m_flow, _cp_f, T_f=_T_f)
                _T_f = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
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
    def from_dimensions(cls, L: ArrayLike, D: ArrayLike, r_b: ArrayLike, x: ArrayLike, y: ArrayLike, R_d: ArrayLike | Callable[[float], Array], fluid: Fluid, basis: Basis, n_segments: int, tilt: float = 0., orientation: float = 0., segment_ratios: ArrayLike | None = None, order: int = 101, order_to_self: int = 21) -> Self:
        """Field of straight boreholes from their dimensions.

        Parameters
        ----------
        L : array_like
            Borehole length (in meters).
        D : array_like
            Borehole buried depth (in meters).
        r_b : array_like
            Borehole radius (in meters).
        x, y : array_like
            Horizontal position (in meters) of the top end of the
            borehole.
        R_d : array_like or callable
            (2, 2,) array of thermal resistances (in m-K/W), or callable
            that takes the mass flow rate as input (in kg/s) and returns a
            (2, 2,) array.
        fluid : fluid
            The fluid.
        basis : basis
            Basis functions.
        n_segments : int
            Number of segments per borehole.
        tilt : array_like, default: ``0.``
            Tilt angle (in radians) of the boreholes with respect to
            vertical.
        orientation : array_like, default: ``0.``
            Orientation (in radians) of the inclined boreholes. An
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

        Returns
        -------
        borefield
            Instance of the `Network` class.

        """
        # Runtime type validation
        if not isinstance(x, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {x}")
        if not isinstance(y, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {y}")
        # Convert input to jax.Array
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        n_boreholes = len(x)
        L = jnp.broadcast_to(L, n_boreholes)
        D = jnp.broadcast_to(D, n_boreholes)
        r_b = jnp.broadcast_to(r_b, n_boreholes)
        tilt = jnp.broadcast_to(tilt, n_boreholes)
        orientation = jnp.broadcast_to(orientation, n_boreholes)
        boreholes = []
        for j in range(n_boreholes):
            path = Path.Line(L[j], D[j], x[j], y[j], tilt[j], orientation[j])
            boreholes.append(SingleUTube(R_d, fluid, r_b[j], path, basis, n_segments, segment_ratios=segment_ratios, order=order, order_to_self=order_to_self))
        return cls(boreholes, fluid)

    @classmethod
    def rectangle_field(cls, N_1: int, N_2: int, B_1: float, B_2: float, L: float, D: float, r_b: float, R_d: ArrayLike | Callable[[float], Array], fluid: Fluid, basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None, order: int = 101, order_to_self: int = 21) -> Self:
        """Field of vertical boreholes in a rectangular configuration.

        Parameters
        ----------
        N_1, N_2 : int
            Number of columns and rows in the borefield.
        B_1, B_2 : float
            Spacing between columns and rows (in meters).
        L : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        R_d : array_like or callable
            (2, 2,) array of thermal resistances (in m-K/W), or callable
            that takes the mass flow rate as input (in kg/s) and returns a
            (2, 2,) array.
        fluid : fluid
            The fluid.
        basis : basis
            Basis functions.
        n_segments : int
            Number of segments per borehole.
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

        Returns
        -------
        borefield
            Instance of the `Network` class.

        """
        # Borehole positions and orientation
        x = jnp.tile(jnp.arange(N_1), N_2) * B_1
        y = jnp.repeat(jnp.arange(N_2), N_1) * B_2
        return cls.from_dimensions(L, D, r_b, x, y, R_d, fluid, basis, n_segments, segment_ratios=segment_ratios, order=order, order_to_self=order_to_self)
        
