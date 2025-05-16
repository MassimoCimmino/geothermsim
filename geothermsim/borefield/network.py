# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import partial
from typing import List, Self, Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..basis import Basis
from .borefield import Borefield
from ..borehole import SingleUTube
from ..path import Path


class Network(Borefield):
    """Borefield with pipes.

    Parameters
    ----------
    boreholes : list of single_u_tube
        Boreholes in the borefield.

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

    def __init__(self, boreholes: List[SingleUTube]):
        super().__init__(boreholes)

    @partial(jit, static_argnames=['self'])
    def effective_borefield_thermal_resistance(self, m_flow: float | Array, cp_f: float) -> float:
        """Effective borefield thermal resistance.

        Parameters
        ----------
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float
            Effective borefield thermal resistance (in m-K/W).

        """
        a = jnp.average(
            self._outlet_fluid_temperature(m_flow, cp_f)[0],
            weights=jnp.broadcast_to(
                m_flow, self.n_boreholes
            )
        )
        b = jnp.sum(self._heat_extraction_rate(self.xi, m_flow, cp_f)[0] * self.w)
        # Effective borehole thermal resistance
        R_field = -0.5 * self.L.sum() * (1. + a) / b
        return R_field

    @partial(jit, static_argnames=['self'])
    def g(self, xi: Array | float, m_flow: float | Array, cp_f: float) -> Array:
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

        Returns
        -------
        a_in : array or float
            (`n_boreholes`, `M,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_boreholes`, M, `n_nodes`,) array of coefficients for the
            borehole wall temperatures.

        """
        return self._heat_extraction_rate(xi, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def g_to_self(self, m_flow: float | Array, cp_f: float) -> Array:
        """Coefficients to evaluate the heat extraction rate at nodes.

        Parameters
        ----------
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : array or float
            (`n_boreholes, `n_nodes`,) array of coefficients for the inlet
            fluid temperature.
        a_b : array
            (`n_boreholes`, `n_nodes`, `n_nodes`,) array of coefficients
            for the borehole wall temperatures.

        """
        return self._heat_extraction_rate(self.xi, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def fluid_temperature(self, xi: Array | float, T_f_in: float, T_b: Array, m_flow: float | Array, cp_f: float) -> Array:
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
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (`n_boreholes`, M, 2,) array of fluid temperatures (in degree
            Celsius).

        """
        a_in, a_b = self._fluid_temperature(xi, m_flow, cp_f)
        T_f = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return T_f

    @partial(jit, static_argnames=['self'])
    def heat_extraction_rate(self, xi: Array | float, T_f_in: float, T_b: Array, m_flow: float | Array, cp_f: float) -> Array:
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
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (`n_boreholes`, M,) array of heat extraction rate (in W/m). If
            `xi` is a ``float``, then ``M=0``.

        """
        a_in, a_b = self._heat_extraction_rate(xi, m_flow, cp_f)
        q = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return q

    @partial(jit, static_argnames=['self'])
    def heat_extraction_rate_to_self(self, T_f_in: float, T_b: Array, m_flow: float | Array, cp_f: float) -> Array:
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
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (`n_boreholes`, `n_nodes`,) array of heat extraction rate
            (in W/m). If `xi` is a ``float``, then ``M=0``.

        """
        return self.heat_extraction_rate(self.xi, T_f_in, T_b, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def outlet_fluid_temperature(self, T_f_in: float, T_b: Array, m_flow: float | Array, cp_f: float) -> Array:
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
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (`n_boreholes`,) array of outlet fluid temperatures (in degree
            Celsius).

        """
        a_in, a_b = self._outlet_fluid_temperature(m_flow, cp_f)
        T_f_out = a_in * T_f_in + vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(a_b, T_b)
        return T_f_out

    def _fluid_temperature(self, xi: Array | float, m_flow: float | Array, cp_f: float) -> Tuple[Array, Array]:
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

        Returns
        -------
        a_in : array
            (`n_boreholes`, M, 2,) array of coefficients for the inlet
            fluid temperature.
        a_b : array
            (`n_boreholes`, M, 2, `n_nodes`,) array of coefficients for
            the borehole wall temperature.

        """
        if len(jnp.shape(m_flow)) == 0 or jnp.shape(m_flow)[0] == 1:
            m_flow = jnp.broadcast_to(
                m_flow / self.n_boreholes,
                self.n_boreholes
            )
        a_in, a_b = zip(*[
            borehole._fluid_temperature(xi, _m_flow, cp_f)
            for borehole, _m_flow in zip(self.boreholes, m_flow)
        ])
        a_in = jnp.stack(a_in, axis=0)
        a_b = jnp.stack(a_b, axis=0)
        return a_in, a_b

    def _heat_extraction_rate(self, xi: Array | float, m_flow: float | Array, cp_f: float) -> Tuple[Array, Array]:
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

        Returns
        -------
        a_in : array or float
            (`n_boreholes`, M,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_boreholes`, M, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        if len(jnp.shape(m_flow)) == 0 or jnp.shape(m_flow)[0] == 1:
            m_flow = jnp.broadcast_to(
                m_flow / self.n_boreholes,
                self.n_boreholes
            )
        a_in, a_b = zip(*[
            borehole._heat_extraction_rate(xi, _m_flow, cp_f)
            for borehole, _m_flow in zip(self.boreholes, m_flow)
        ])
        a_in = jnp.stack(a_in, axis=0)
        a_b = jnp.stack(a_b, axis=0)
        return a_in, a_b

    def _outlet_fluid_temperature(self, m_flow: float | Array, cp_f: float) -> Tuple[Array, Array]:
        """Coefficients to evaluate the outlet fluid temperatures.

        Parameters
        ----------
        m_flow : float or array
            Total fluid mass flow rate (in kg/s), or (`n_boreholes`,)
            array of fluid mass flow rate per borehole.
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : array
            (`n_boreholes`,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_boreholes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        if len(jnp.shape(m_flow)) == 0 or jnp.shape(m_flow)[0] == 1:
            m_flow = jnp.broadcast_to(
                m_flow / self.n_boreholes,
                self.n_boreholes
            )
        a_in, a_b = zip(*[
            borehole._outlet_fluid_temperature(_m_flow, cp_f)
            for borehole, _m_flow in zip(self.boreholes, m_flow)
        ])
        a_in = jnp.stack(a_in, axis=0)
        a_b = jnp.stack(a_b, axis=0)
        return a_in, a_b

    @classmethod
    def from_dimensions(cls, L: ArrayLike, D: ArrayLike, r_b: ArrayLike, x: ArrayLike, y: ArrayLike, R_d: ArrayLike | Callable[[float], Array], basis: Basis, n_segments: int, tilt: float = 0., orientation: float = 0., segment_ratios: ArrayLike | None = None, order: int | None = None) -> Self:
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
            path = Path.from_dimensions(L[j], D[j], x[j], y[j], tilt[j], orientation[j])
            boreholes.append(SingleUTube(R_d, r_b[j], path, basis, n_segments, segment_ratios=segment_ratios))
        return cls(boreholes)

    @classmethod
    def rectangle_field(cls, N_1: int, N_2: int, B_1: float, B_2: float, L: float, D: float, r_b: float, R_d: ArrayLike | Callable[[float], Array], basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None, order: int | None = None) -> Self:
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
        basis : basis
            Basis functions.
        n_segments : int
            Number of segments per borehole.
        segment_ratios : array_like or None, default: None
            Normalized size of the segments. Should total ``1``
            (i.e. ``sum(segment_ratios) = 1``). If `segment_ratios` is
            ``None``, segments of equal size are considered (i.e.
            ``segment_ratios[v] = 1 / n_segments``).

        Returns
        -------
        borefield
            Instance of the `Network` class.

        """
        # Borehole positions and orientation
        x = jnp.tile(jnp.arange(N_1), N_2) * B_1
        y = jnp.repeat(jnp.arange(N_2), N_1) * B_2
        return cls.from_dimensions(L, D, r_b, x, y, R_d, basis, n_segments, segment_ratios=segment_ratios)
        
