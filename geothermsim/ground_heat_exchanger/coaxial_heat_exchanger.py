# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ._ground_heat_exchanger import _GroundHeatExchanger
from ..fluid import Fluid
from ..thermal_resistance import (
conduction_thermal_resistance_circular_pipe,
convective_heat_transfer_coefficient_annular_pipe,
convective_heat_transfer_coefficient_circular_pipe,
Multipole
)


class CoaxialHeatExchanger(_GroundHeatExchanger):
    """Coaxial heat exchanger cross-section.

    Parameters
    ----------
    p : array_like
        (`n_pipes` / 2, 2) array of pipe positions (in meters). The
        position ``(0, 0)`` corresponds to the axis of the borehole.
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
    R_p_inner : array
        Conduction thermal resistance of the inner pipes (in m-K/W).
    R_p_outer : array
        Conduction thermal resistance of the outer pipes (in m-K/W).

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
        r_p_in = jnp.asarray(r_p_in)
        r_p_out = jnp.asarray(r_p_out)
        k_p = jnp.broadcast_to(k_p, 2 * p.shape[0])
        p = jnp.atleast_2d(p)

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
        self._n_pipes_over_two = p.shape[0]
        self.n_pipes = 2 * self._n_pipes_over_two
        if self.parallel:
            self._m_flow_factor = 1 / self._n_pipes_over_two
        else:
            self._m_flow_factor = 1

        # --- Identify inner and outer pipes ---
        indices = jnp.arange(self.n_pipes, dtype=int).reshape(2, -1)
        indices_inner = jnp.argmin(r_p_in.reshape(2, -1), axis=0)
        self.indices_inner = indices[indices_inner, indices[0]]
        indices_outer = jnp.argmax(r_p_in.reshape(2, -1), axis=0)
        self.indices_outer = indices[indices_outer, indices[0]]
        self._indices_outer_mesh = jnp.meshgrid(
            self.indices_outer,
            self.indices_outer,
            indexing='ij'
        )
        # Inner and outer pipe radii
        self._r_p_in_inner = r_p_in[self.indices_inner]
        self._r_p_out_inner = r_p_out[self.indices_inner]
        self._r_p_in_outer = r_p_in[self.indices_outer]
        self._r_p_out_outer = r_p_out[self.indices_outer]
        # Conduction thermal resistance through pipe walls
        self.R_p_inner = vmap(
            conduction_thermal_resistance_circular_pipe,
            in_axes=(0, 0, 0),
            out_axes=0
        )(self._r_p_in_inner, self._r_p_out_inner, self.k_p[self.indices_inner])
        self.R_p_outer = vmap(
            conduction_thermal_resistance_circular_pipe,
            in_axes=(0, 0, 0),
            out_axes=0
        )(self._r_p_in_outer, self._r_p_out_outer, self.k_p[self.indices_outer])

        # --- Multipole model ---
        self.multipole = Multipole(
            self.r_b, self._r_p_out_outer, self.p, self.k_s, self.k_b,
            J=self.J)

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
        h_fluid_inner = vmap(
            convective_heat_transfer_coefficient_circular_pipe,
            in_axes=(None, 0, 0, 0, 0, 0, None),
            out_axes=0
        )(
            m_flow_pipe,
            self._r_p_in_inner,
            mu_f[self.indices_inner],
            rho_f[self.indices_inner],
            k_f[self.indices_inner],
            Pr[self.indices_inner],
            self.epsilon
        )
        R_f_inner = 1 / (2 * jnp.pi * self._r_p_in_inner * h_fluid_inner)
        # Convection thermal resistances in annular pipes
        h_fluid_outer = vmap(
            convective_heat_transfer_coefficient_annular_pipe,
            in_axes=(None, 0, 0, 0, 0, 0, 0, None),
            out_axes=0
        )(
            m_flow_pipe,
            self._r_p_out_inner,
            self._r_p_in_outer,
            mu_f[self.indices_outer],
            rho_f[self.indices_outer],
            k_f[self.indices_outer],
            Pr[self.indices_outer], 
            self.epsilon
        )
        R_f_in_outer = 1 / (2 * jnp.pi * self._r_p_out_inner * h_fluid_outer[:, 0])
        R_f_out_outer = 1 / (2 * jnp.pi * self._r_p_in_outer * h_fluid_outer[:, 1])
        # Short-circuit thermal resistances
        R_ff = R_f_inner + R_f_in_outer + self.R_p_inner
        # Delta-circuit thermal resistances through grout
        R_fp = R_f_out_outer + self.R_p_outer
        R_d_grout = self.multipole.thermal_resistances(R_fp)
        # Full delta-circuit of thermal resistances
        R_d = jnp.full((self.n_pipes, self.n_pipes), jnp.inf)
        R_d = R_d.at[self._indices_outer_mesh].set(R_d_grout)
        R_d = R_d.at[self.indices_outer, self.indices_inner].set(R_ff)
        R_d = R_d.at[self.indices_inner, self.indices_outer].set(R_ff)
        return R_d
