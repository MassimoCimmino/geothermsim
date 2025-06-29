# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import switch

from .darcy_friction_factor import darcy_friction_factor
from .reynolds_number import (
reynolds_number_annular_pipe,
reynolds_number_circular_pipe
)


def convective_heat_transfer_coefficient_annular_pipe(m_flow: float, r_p_in: float, r_p_out: float, mu_f: float, rho_f: float, k_f: float, Pr: float, epsilon: float) -> Array:
    """Convective heat transfer coefficient in circular pipe.

    Parameters
    ----------
    m_flow : float
        Fluid mass flow rate (in kg/s).
    r_p_in : float
        Outer radius of the inner pipe (in meters).
    r_p_out : float
        Inner radius of the outer pipe (in meters).
    mu_f : float
        Fluid dynamic viscosity (in kg/m-s).
    rho_f : float
        Fluid density (in kg/m3).
    k_f : float
        Fluid thermal conductivity (in W/m-K).
    Pr : float
        Prandtl number.
    epsilon : float
        Pipe surface roughness (in meters).

    Returns
    -------
    array
        Convective heat transfer coefficients at the inner and outer
        surfaces of the annulus (in W/m2-K).
    
    """
    # Reynolds number
    Re = reynolds_number_annular_pipe(m_flow, r_p_in, r_p_out, mu_f, rho_f)
    # Darcy friction factor
    D = 2 * (r_p_out - r_p_in)
    E = epsilon / D
    f_darcy = darcy_friction_factor(Re, E)
    # Nusselt number
    r_ratio = r_p_in / r_p_out
    Nu = nusselt_number_annular_pipe(Re, Pr, f_darcy, r_ratio)
    # Convective heat transfer coefficient
    h_fluid = k_f * Nu / D
    return h_fluid


def convective_heat_transfer_coefficient_circular_pipe(m_flow: float, r_p: float, mu_f: float, rho_f: float, k_f: float, Pr: float, epsilon: float) -> float:
    """Convective heat transfer coefficient in circular pipe.

    Parameters
    ----------
    m_flow : float
        Fluid mass flow rate (in kg/s).
    r_p : float
        Inner radius of the pipe (in meters).
    mu_f : float
        Fluid dynamic viscosity (in kg/m-s).
    rho_f : float
        Fluid density (in kg/m3).
    k_f : float
        Fluid thermal conductivity (in W/m-K).
    Pr : float
        Prandtl number.
    epsilon : float
        Pipe surface roughness (in meters).

    Returns
    -------
    float
        Convective heat transfer coefficient (in W/m2-K).
    
    """
    # Reynolds number
    Re = reynolds_number_circular_pipe(m_flow, r_p, mu_f, rho_f)
    # Darcy friction factor
    D = 2 * r_p
    E = epsilon / D
    f_darcy = darcy_friction_factor(Re, E)
    # Nusselt number
    Nu = nusselt_number_circular_pipe(Re, Pr, f_darcy)
    # Convective heat transfer coefficient
    h_fluid = k_f * Nu / D
    return h_fluid


def nusselt_number_annular_pipe(Re: float, Pr: float, f_darcy: float, r_ratio: float) -> Array:
    """Nusselt numbers in annular pipe.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Pr : float
        Prandtl number.
    f_darcy : float
        Darcy friction factor.
    r_ratio : float
        Ratio of inner and outer radii or the annulus.

    Returns
    -------
    array
        Nusselt number at the inner and outer pipe surfaces.

    """
    Re_crit_lower = 2300.
    Re_crit_upper = 4000.
    # Nusselt number for laminar flow
    def _nusselt_number_laminar_flow(_Re: float) -> Array:
        Nu_laminar = 3.66 + 1.2 * r_ratio**jnp.array([-0.8, 0.5])
        return Nu_laminar
    # Nusselt number for turbulent flow
    def _nusselt_number_turbulent_flow(_Re: float) -> Array:
        Nu_turbulent = nusselt_number_turbulent_flow(_Re, Pr, f_darcy)
        return jnp.broadcast_to(Nu_turbulent, 2)
    # Nusselt number in transition regime
    Nu_crit_lower = _nusselt_number_laminar_flow(Re_crit_lower)
    Nu_crit_upper = _nusselt_number_turbulent_flow(Re_crit_upper)
    def _nusselt_number_transition(_Re: float) -> Array:
        Nu_transition = nusselt_number_transition(
            _Re, Re_crit_lower, Re_crit_upper, Nu_crit_lower, Nu_crit_upper)
        return Nu_transition
    # Piecewise evaluation of Nusselt number
    funclist = [
        _nusselt_number_laminar_flow,
        _nusselt_number_transition,
        _nusselt_number_turbulent_flow
    ]
    bounds = jnp.array([Re_crit_lower, Re_crit_upper])
    index = jnp.searchsorted(bounds, Re)
    Nu = switch(index, funclist, Re)
    return Nu


def nusselt_number_circular_pipe(Re: float, Pr: float, f_darcy: float) -> float:
    """Nusselt number in circular pipe.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Pr : float
        Prandtl number.
    f_darcy : float
        Darcy friction factor.

    Returns
    -------
    float
        Nusselt number.

    """
    Re_crit_lower = 2300.
    Re_crit_upper = 4000.
    # Condition on the flow regime
    condlist = [Re <= Re_crit_lower, Re >= Re_crit_upper]
    # Nusselt number for laminar flow
    Nu_laminar = 3.66
    # Nusselt number for turbulent flow
    _nusselt_number_turbulent_flow = partial(
        nusselt_number_turbulent_flow,
        Pr=Pr,
        f_darcy=f_darcy)
    # Nusselt number at the onset of turbulent flow
    Nu_crit_upper = nusselt_number_turbulent_flow(
        Re_crit_upper, Pr, f_darcy)
    # Nusselt number in transition regime
    _nusselt_number_transition = partial(
        nusselt_number_transition,
        Re_crit_lower=Re_crit_lower,
        Re_crit_upper=Re_crit_upper,
        Nu_crit_lower=Nu_laminar,
        Nu_crit_upper=Nu_crit_upper)
    # Piecewise evaluation of Nusselt number
    funclist = [Nu_laminar, _nusselt_number_turbulent_flow, _nusselt_number_transition]
    Nu = jnp.piecewise(jnp.float32(Re), condlist, funclist)
    return Nu


def nusselt_number_transition(Re: float, Re_crit_lower: float, Re_crit_upper: float, Nu_crit_lower: float | Array, Nu_crit_upper: float | Array) -> float | Array:
    """Nusselt number for turbulent flow in any pipe.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Re_crit_lower : float
        Reynolds number at the lower limit of the transition regime.
    Re_crit_upper : float
        Reynolds number at the upper limit of the transition regime.
    Nu_crit_lower : float
        Nusselt number at the lower limit of the transition regime.
    Nu_crit_upper : float
        Nusselt number at the upper limit of the transition regime.

    Returns
    -------
    float
        Nusselt number.

    """
    gamma = (Re - Re_crit_lower) / (Re_crit_upper - Re_crit_lower)
    Nu = (1 - gamma) * Nu_crit_lower + gamma * Nu_crit_upper
    return Nu


def nusselt_number_turbulent_flow(Re: float, Pr: float, f_darcy: float) -> float:
    """Nusselt number for turbulent flow in any pipe.

    Parameters
    ----------
    Re : float
        Reynolds number.
    Pr : float
        Prandtl number.
    f_darcy : float
        Darcy friction factor.

    Returns
    -------
    float
        Nusselt number.

    """
    Nu = 0.125 * f_darcy * (Re - 1.0e3) * Pr / (
        1 + 12.7 * jnp.sqrt(0.125 * f_darcy) * (Pr**(2 / 3) - 1)
    )
    return Nu
