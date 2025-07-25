# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import partial
from typing_extensions import Self

from interpax import Interpolator1D
from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from scp.water import Water
from scp.ethyl_alcohol import EthylAlcohol
from scp.ethylene_glycol import EthyleneGlycol
from scp.methyl_alcohol import MethylAlcohol
from scp.propylene_glycol import PropyleneGlycol


class Fluid:
    """Heat carrier fluid.

    Parameters
    ----------
    T_nominal : float
        Nominal fluid temperature (in degree Celsius).
    f_conductivity : callable
        Function that takes a temperature or an (N,) array of
        temperatures ``T`` and returns a float or an (N,) array of
        fluid thermal conductivities.
    f_density : callable
        Function that takes a temperature or an (N,) array of
        temperatures ``T`` and returns a float or an (N,) array of
        fluid densities.
    f_specific_heat : callable
        Function that takes a temperature or an (N,) array of
        temperatures ``T`` and returns a float or an (N,) array of
        fluid specific heat capacities.
    f_viscosity : callable
        Function that takes a temperature or an (N,) array of
        temperatures ``T`` and returns a float or an (N,) array of
        fluid dynamic viscosities.
    f_prandtl : callable, default: None
        Function that takes a temperature or an (N,) array of
        temperatures ``T`` and returns a float or an (N,) array of
        fluid Prandtl numbers. If ``None``, the Prandtl numbers are
        instead evaluated from the other physical properties.

    """

    def __init__(
        self,
        T_nominal: float,
        f_conductivity: Callable[[float| Array], float| Array],
        f_density: Callable[[float| Array], float| Array],
        f_specific_heat: Callable[[float| Array], float| Array],
        f_viscosity: Callable[[float| Array], float| Array],
        f_prandtl: Callable[[float| Array], float| Array] | None = None
    ):
        if f_prandtl is None:
            def f_prandtl(_T: float | Array) -> float | Array:
                """Prandtl number."""
                cp_f = f_specific_heat(_T)
                mu_f = f_viscosity(_T)
                k_f = f_conductivity(_T)
                return cp_f * mu_f / k_f

        # --- Class atributes ---
        self.T_nominal = T_nominal
        self._k_f = f_conductivity(T_nominal)
        self._rho_f = f_density(T_nominal)
        self._cp_f = f_specific_heat(T_nominal)
        self._mu_f = f_viscosity(T_nominal)
        self._prandtl = f_prandtl(T_nominal)

        # --- Fluid properties functions ---
        self.f_conductivity = f_conductivity
        self.f_density = f_density
        self.f_specific_heat = f_specific_heat
        self.f_viscosity = f_viscosity
        self.f_prandtl = f_prandtl

    def conductivity(self, T: float | Array | None = None) -> float | Array:
        """Thermal conductivity.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The thermal conductivity (in W/m-K).

        """
        if T is None:
            return self._k_f
        return self.f_conductivity(T)

    def density(self, T: float | Array | None = None) -> float | Array:
        """Density.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The density (in kg/m3).

        """
        if T is None:
            return self._rho_f
        return self.f_density(T)

    def prandtl(self, T: float | Array | None = None) -> float | Array:
        """Prandtl number.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The Prandtl number.

        """
        if T is None:
            return self._prandtl
        return self.f_prandtl(T)

    def specific_heat(self, T: float | Array | None = None) -> float | Array:
        """Specific isobaric heat capacity.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The specific heat capacity (in J/kg-K).

        """
        if T is None:
            return self._cp_f
        return self.f_specific_heat(T)

    def viscosity(self, T: float | Array | None = None) -> float | Array:
        """Dynamic viscosity.

        Parameters
        ----------
        T : float, array or None, default: None
            Fluid temperature (in degree Celsius). If ``None``, the property
            at `T_nominal` is returned.

        Returns
        -------
        float or array
            The dynamic viscosity (in Pa-s).

        """
        if T is None:
            return self._mu_f
        return self.f_viscosity(T)

    @classmethod
    def from_scp(cls, fluid_name: str, T_nominal: float, x: float | None = None, method: str = 'cubic', num: int = 51) -> Self:
        """Import heat carrier fluid from SecondaryCoolantProps.

        Parameters
        ----------
        fluid_name : {'Water', 'EthylAlcohol', 'EthyleneGlycol', 'MethylAlcohol', 'PropyleneGlycol'}
            Name of the fluid, as per the nomenclature of
            ``SecondaryCoolantProps``.
        T_nominal : float
            Nominal fluid temperature (in degree Celsius).
        x : float or None, default: None
            Mass fraction of the brine (from ``0.`` to ``1.``). Not
            used for fluid ``'Water'``.
        method : str, default: 'cubic'
            Interpolation methods to be used by
            ``interpax.Interpolator1D`` for the fluid properties.
        num : int, default: 51
            Number of temperature values at which to evaluate fluid
            physical properties and create the inetrpolator. The minimum
            and maximum temperatures are chosen automatically based on
            the fluid and its mass fraction.

        Returns
        -------
        fluid
            The fluid.

        """
        fluid_lower = fluid_name.lower()
        if fluid_lower == 'water':
            fluid = Water()
        elif x is None:
            raise TypeError(f"Expected float input; got x = {x}")
        elif fluid_lower == 'ethylalcohol' or fluid_lower == 'ethyl_alcohol':
            fluid = EthylAlcohol(x)
        elif fluid_lower == 'ethyleneglycol' or fluid_lower == 'ethylene_glycol':
            fluid = EthyleneGlycol(x)
        elif fluid_lower == 'methylalcohol' or fluid_lower == 'methyl_alcohol':
            fluid = MethylAlcohol(x)
        elif fluid_lower == 'propyleneglycol' or fluid_lower == 'propylene_glycol':
            fluid = PropyleneGlycol(x)
        else:
            raise ValueError(f"Fluid {fluid_lower} is not a valid SecondaryCoolantProps fluid.")

        # Find minimum and maximum fluid temperatures
        T_min = fluid.t_min
        T_max = fluid.t_max

        # Evaluate properties at evenly spaced temperatures
        T = jnp.linspace(T_min, T_max, num=num)
        k_f = jnp.array([fluid.conductivity(_T) for _T in T])
        rho_f = jnp.array([fluid.density(_T) for _T in T])
        prandtl = jnp.array([fluid.prandtl(_T) for _T in T])
        cp_f = jnp.array([fluid.specific_heat(_T) for _T in T])
        mu_f = jnp.array([fluid.viscosity(_T) for _T in T])

        # Create interpolators
        extrap_indices = jnp.array([0, -1])
        f_conductivity = Interpolator1D(T, k_f, method=method, extrap=k_f[extrap_indices])
        f_density = Interpolator1D(T, rho_f, method=method, extrap=rho_f[extrap_indices])
        f_prandtl = Interpolator1D(T, prandtl, method=method, extrap=prandtl[extrap_indices])
        f_specific_heat = Interpolator1D(T, cp_f, method=method, extrap=cp_f[extrap_indices])
        f_viscosity = Interpolator1D(T, mu_f, method=method, extrap=mu_f[extrap_indices])
        return cls(T_nominal, f_conductivity, f_density, f_specific_heat, f_viscosity, f_prandtl=f_prandtl)
