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
from ..path import Path


class _Tube(Borehole, ABC):
    """Geothermal borehole with tube.

    Parameters
    ----------
    R_d : array_like or callable
        (`n_pipes`, `n_pipes`,) array of thermal resistances (in m-K/W),
        or callable that takes the mass flow rate as input (in kg/s) and
        returns a (`n_pipes`, `n_pipes`,) array.
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

    def __init__(self, R_d: ArrayLike | Callable[[float], Array], r_b: float, path: Path, basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None, parallel: bool = True):
        # Runtime type validation
        if not isinstance(R_d, ArrayLike) and not callable(R_d):
            raise TypeError(f"Expected arraylike or callable input; got {R_d}")
        # Convert input to jax.Array
        if isinstance(R_d, ArrayLike):
            R_d = jnp.atleast_2d(R_d)
            def thermal_resistances(m_flow: float) -> Array:
                return R_d
            self.thermal_resistances = thermal_resistances
        else:
            self.thermal_resistances = R_d

        # Initialize borehole
        super().__init__(r_b, path, basis, n_segments, segment_ratios=segment_ratios)
        # Other attributes
        self.R_d = R_d
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
            self._top_connectivity_in = jnp.ones(self._n_pipes_over_two)
            self._top_connectivity_u = jnp.zeros(
                (self._n_pipes_over_two, self._n_pipes_over_two)
            )
            # Coefficient for mixing of the fluid at the outlet
            # T_f_out = m_u @ T_f_u(-1)
            self._mixing_u = jnp.full(
                self._n_pipes_over_two,
                self._m_flow_factor
            )
        else:
            # Factor for flow division amongst pipes
            self._m_flow_factor = 1
            # Coefficients for connectivity at the top of the borehole
            # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
            self._top_connectivity_in = jnp.concatenate([
                jnp.ones(1),
                jnp.zeros(self._n_pipes_over_two - 1),
            ])
            self._top_connectivity_u = jnp.eye(
                self._n_pipes_over_two,
                k=-1
            )
            # Coefficient for mixing of the fluid at the outlet
            # T_f_out = m_u @ T_f_u(-1)
            self._mixing_u = jnp.concatenate([
                jnp.zeros(self._n_pipes_over_two - 1),
                jnp.ones(1),
            ])

    @partial(jit, static_argnames=['self'])
    def effective_borehole_thermal_resistance(self, m_flow: float, cp_f: float) -> float:
        """Effective borehole thermal resistance.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float
            Effective borehole thermal resistance (in m-K/W).

        """
        beta_ij = self._beta_ij(m_flow, cp_f)
        a = self._outlet_fluid_temperature_a_in(beta_ij)
        b = self._heat_extraction_rate_a_in(self.xi, m_flow, cp_f, beta_ij) @ self.w
        # Effective borehole thermal resistance
        R_b = -0.5 * self.L * (1. + a) / b
        return R_b

    @partial(jit, static_argnames=['self'])
    def g(self, xi: Array | float, m_flow: float, cp_f: float) -> Tuple[Array | float, Array]:
        """Coefficients to evaluate the heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : array or float
            (M,) array of coefficients for the inlet fluid temperature.
        a_b : array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        return self._heat_extraction_rate(xi, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def g_to_self(self, m_flow: float, cp_f: float) -> Tuple[Array, Array]:
        """Coefficients to evaluate the heat extraction rate at nodes.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : array or float
            (`n_nodes`,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (`n_nodes`, `n_nodes`,) array of coefficients for the borehole
            wall temperature.

        """
        return self._heat_extraction_rate(self.xi, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def fluid_temperature(self, xi: Array | float, T_f_in: float, T_b: Array, m_flow: float, cp_f: float) -> Array:
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
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (M, 2,) array of fluid temperatures (in degree Celsius).

        """
        a_in, a_b = self._fluid_temperature(xi, m_flow, cp_f)
        T_f = a_in * T_f_in + a_b @ T_b
        return T_f

    @partial(jit, static_argnames=['self'])
    def heat_extraction_rate(self, xi: Array | float, T_f_in: float, T_b: Array, m_flow: float, cp_f: float) -> Array | float:
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
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (M,) array of heat extraction rate (in W/m). If `xi` is a
            ``float``, then ``M=0``.

        """
        a_in, a_b = self._heat_extraction_rate(xi, m_flow, cp_f)
        q = a_in * T_f_in + a_b @ T_b
        return q

    @partial(jit, static_argnames=['self'])
    def heat_extraction_rate_to_self(self, T_f_in: float, T_b: Array, m_flow: float, cp_f: float) -> Array:
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
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (`n_nodes`,) array of heat extraction rate (in W/m). If `xi`
            is a ``float``, then ``M=0``.

        """
        return self.heat_extraction_rate(self.xi, T_f_in, T_b, m_flow, cp_f)

    @partial(jit, static_argnames=['self'])
    def outlet_fluid_temperature(self, T_f_in: float, T_b: Array, m_flow: float, cp_f: float) -> float:
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
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        float
            Outlet fluid temperature (in degree Celsius).

        """
        a_in, a_b = self._outlet_fluid_temperature(m_flow, cp_f)
        T_f_out = a_in * T_f_in + a_b @ T_b
        return T_f_out

    def _fluid_temperature(self, xi: Array | float, m_flow: float, cp_f: float) -> Tuple[Array, Array]:
        """Coefficients to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : array
            (M, `n_pipes`,) array of coefficients for the inlet fluid
            temperature.
        a_b : array
            (M, `n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        beta_ij = self._beta_ij(m_flow, cp_f)
        a_in = self._fluid_temperature_a_in(xi, beta_ij)
        a_b = self._fluid_temperature_a_b(xi, beta_ij)
        return a_in, a_b

    @abstractmethod
    def _fluid_temperature_a_in(self, xi: Array | float, beta_ij: Array) -> Array:
        """Inlet coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (M, `n_pipes`,) array of coefficients for the inlet fluid
            temperature.

        """
        ...

    @abstractmethod
    def _fluid_temperature_a_b(self, xi: Array | float, beta_ij: Array) -> Array:
        """Borehole wall coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        ...

    def _heat_extraction_rate(self, xi: Array | float, m_flow: float, cp_f: float) -> Tuple[Array | float, Array]:
        """Coefficients to evaluate the heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : array or float
            (M,) array of coefficients for the inlet fluid temperature.
        a_b : array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        beta_ij = self._beta_ij(m_flow, cp_f)
        a_in = self._heat_extraction_rate_a_in(xi, m_flow, cp_f, beta_ij)
        a_b = self._heat_extraction_rate_a_b(xi, m_flow, cp_f, beta_ij)
        return a_in, a_b

    def _heat_extraction_rate_a_in(self, xi: Array | float, m_flow: float, cp_f: float, beta_ij: Array) -> Array | float:
        """Inlet coefficient to evaluate the heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array or float
            (M,) array of coefficients for the inlet fluid temperature.

        """
        b_in = self._fluid_temperature_a_in(xi, beta_ij)
        R_d = 1 / (self._m_flow_factor * m_flow * cp_f * beta_ij)
        a_in = -(b_in / jnp.diag(R_d)).sum(axis=1)
        return a_in

    def _heat_extraction_rate_a_b(self, xi: Array | float, m_flow: float, cp_f: float, beta_ij: Array) -> Array:
        """Borehole wall coefficient to evaluate the heat extraction rate.

        Parameters
        ----------
        xi : array or float
            (M,) array of coordinates along the borehole.
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance coefficients.

        Returns
        -------
        array
            (M, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        b_b = self._fluid_temperature_a_b(xi, beta_ij)
        R_d = 1 / (self._m_flow_factor * m_flow * cp_f * beta_ij)
        R_b = 1 / (1 / jnp.diag(R_d)).sum()
        a_b = -(b_b / jnp.diag(R_d)[:, None]).sum(axis=1) + vmap(self.f_psi, in_axes=0)(xi) / R_b
        return a_b

    def _outlet_fluid_temperature(self, m_flow: float, cp_f: float) -> Tuple[float, Array]:
        """Coefficients to evaluate the outlet fluid temperature.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        a_in : float
            Coefficient for the inlet fluid temperature.
        a_b : array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        beta_ij = self._beta_ij(m_flow, cp_f)
        a_in = self._outlet_fluid_temperature_a_in(beta_ij)
        a_b = self._outlet_fluid_temperature_a_b(beta_ij)
        return a_in, a_b

    @abstractmethod
    def _outlet_fluid_temperature_a_in(self, beta_ij: Array) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        float
            Coefficient for the inlet fluid temperature.

        """
        ...

    @abstractmethod
    def _outlet_fluid_temperature_a_b(self, beta_ij: Array) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        ...

    def _beta_ij(self, m_flow: float, cp_f: float) -> Array:
        """Thermal conductance coefficients.

        Parameters
        ----------
        m_flow : float
            Fluid mass flow rate (in kg/s).
        cp_f : float
            Fluid specific isobaric heat capacity (in J/kg-K).

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        """
        return 1. / (self._m_flow_factor * m_flow * cp_f * self.thermal_resistances(m_flow))

    @classmethod
    def from_dimensions(cls, R_d: ArrayLike | Callable[[float], Array], L: float, D: float, r_b: float, x: float, y: float, basis: Basis, n_segments: int, tilt: float = 0., orientation: float = 0., segment_ratios: ArrayLike | None = None, parallel: bool = True) -> Self:
        """Straight borehole from its dimensions.

        Parameters
        ----------
        R_d : array_like or callable
            (`n_pipes`, `n_pipes`,) array of thermal resistances
            (in m-K/W), or callable that takes the mass flow rate as input
            (in kg/s) and returns a (`n_pipes`, `n_pipes`,) array.
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
        parallel : bool, default: True
            True if pipes are in parallel. False if pipes are in series.

        Returns
        -------
        borehole
            Instance of the `Borehole` class.

        """
        path = Path.from_dimensions(L, D, x, y, tilt, orientation)
        return cls(R_d, r_b, path, basis, n_segments, segment_ratios=segment_ratios, parallel=parallel)
