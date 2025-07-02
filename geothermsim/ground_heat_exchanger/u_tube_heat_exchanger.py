# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ._ground_heat_exchanger import _GroundHeatExchanger
from ..fluid import Fluid
from ..thermal_resistance import (
conduction_thermal_resistance_circular_pipe,
convective_heat_transfer_coefficient_circular_pipe,
Multipole
)


class UTubeHeatExchanger(_GroundHeatExchanger):
    """U-tube heat exchanger cross-section.

    Parameters
    ----------
    p : array_like
        (`n_pipes`, 2) array of pipe positions (in meters). The position
        ``(0, 0)`` corresponds to the axis of the borehole.
    r_p_in : array_like
        Inner radius of the pipes (in meters).
    r_p_out : array_like
        Outer radius of the pipes (in meters).
    r_b : float
        Borehole radius (in meters).
    fluid : fluid
        The fluid.
    k_s : float
        Ground thermal conductivity (in W/m-K).
    k_b : float
        Grout thermal conductivity (in W/m-K).
    k_p : array_like
        Thermal conductivity of the pipes (in W/m-K).
    epsilon : float
        Pipe surface roughness (in meters).
    parallel : bool, default: True
        True if pipes are in parallel. False if pipes are in series.
    J : int, default: 3
        Order of the multipole solution.

    Attributes
    ----------
    n_pipes : int
        Number of pipes.
    R_p : array
        Conduction thermal resistance (in m-K/W).

    """

    def __init__(self, p: ArrayLike, r_p_in: ArrayLike, r_p_out: ArrayLike, r_b: float, fluid: Fluid, k_s: float, k_b: float, k_p: ArrayLike, epsilon: float, parallel: bool = True, J: int = 3):
        # Runtime type validation
        if not isinstance(r_p_in, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {r_p_in}")
        if not isinstance(r_p_out, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {r_p_out}")
        if not isinstance(k_p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {k_p}")
        if not isinstance(p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {p}")
        # Convert input to jax.Array
        p = jnp.atleast_2d(p)
        r_p_in = jnp.broadcast_to(r_p_in, p.shape[0])
        r_p_out = jnp.broadcast_to(r_p_out, p.shape[0])
        k_p = jnp.broadcast_to(k_p, p.shape[0])

        # --- Class atributes ---
        # Parameters
        self.p = p
        self.r_p_in = r_p_in
        self.r_p_out = r_p_out
        self.r_b = r_b
        self.fluid = fluid
        self.k_s = k_s
        self.k_b = k_b
        self.k_p = k_p
        self.epsilon = epsilon
        self.parallel = parallel
        self.J = J
        # Additional attributes
        self.n_pipes = p.shape[0]
        self.R_p = vmap(
            conduction_thermal_resistance_circular_pipe,
            in_axes=(0, 0, 0),
            out_axes=0
        )(self.r_p_in, self.r_p_out, self.k_p)
        self._n_pipes_over_two = int(self.n_pipes / 2)
        if self.parallel:
            self._m_flow_factor = 1 / self._n_pipes_over_two
        else:
            self._m_flow_factor = 1

        # --- Multipole model ---
        self.multipole = Multipole(
            self.r_b, self.r_p_out, self.p, self.k_s, self.k_b, J=self.J)

    @partial(jit, static_argnames=['self'])
    def thermal_resistances(self, m_flow: float, T_f: float | Array | None = None) -> Array:
        """Evaluate delta-circuit thermal resistances.

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
        array
            (`n_pipes`, `n_pipes`,) array of delta-circuit thermal
            resistances (in m-K/W).

        """
        m_flow_pipe = self._m_flow_factor * m_flow
        # Fluid physical properties
        rho_f = jnp.broadcast_to(
            self.fluid.density(T_f),
            self.n_pipes
        )
        mu_f = jnp.broadcast_to(
            self.fluid.viscosity(T_f),
            self.n_pipes
        )
        k_f = jnp.broadcast_to(
            self.fluid.conductivity(T_f),
            self.n_pipes
        )
        Pr = jnp.broadcast_to(
            self.fluid.prandtl(T_f),
            self.n_pipes
        )
        # Convection thermal resistance in circular pipes
        h_fluid = vmap(
            convective_heat_transfer_coefficient_circular_pipe,
            in_axes=(None, 0, 0, 0, 0, 0, None),
            out_axes=0
        )(
            m_flow_pipe,
            self.r_p_in,
            mu_f,
            rho_f,
            k_f,
            Pr,
            self.epsilon
        )
        R_f = 1 / (2 * jnp.pi * self.r_p_in * h_fluid)
        # Delta-circuit thermal resistances through grout
        R_fp = R_f + self.R_p
        R_d = self.multipole.thermal_resistances(R_fp)
        return R_d
