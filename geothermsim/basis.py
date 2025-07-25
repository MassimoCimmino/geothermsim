# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing import Tuple
from typing_extensions import Self

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from scipy.special import roots_legendre

from .utilities import quad


class Basis:
    """Polynomial basis functions.

    Parameters
    ----------
    xi : array_like
        (`n_nodes`,) array of node coordinates along the interval
        ``[-1, 1]``.
    w : array_like or None, default: None
        Quadrature weights for numerical integration using `xi` as
        quadrature points. If `w` is ``None``, the weights are evaluated
        through numerical integration of order `order`.

    Attributes
    ----------
    n_nodes : int
        Number of nodes.

    """

    def __init__(self, xi: ArrayLike, w: ArrayLike | None = None):
        # Runtime type validation
        if not isinstance(xi, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {xi}")
        # Convert input to jax.Array
        xi = jnp.asarray(xi)

        # --- Class atributes ---
        self.xi = xi
        self.n_nodes = len(xi)

        # --- Basis functions (psi) ---
        self._psi_coefs = jnp.polyfit(
            self.xi,
            jnp.eye(self.n_nodes),
            self.n_nodes - 1)

        # Integration weights
        if w is None:
            w = quad(
                self.f_psi,
                order=self.n_nodes
            )
        self.w = w

    def f(self, xi: float | Array, f_nodes: Array) -> float | Array:
        """Value at coordinate from values at nodes.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates.
        f_nodes : array
            (`n_nodes`,) array of values at nodes.

        Returns
        -------
        float or array
            Value or (M,) array of values at requested coordinates
            evaluated using polynomial basis functions.

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(self.f, in_axes=(0, None), out_axes=0)(xi, f_nodes)
        return self._f(xi, f_nodes, self._psi_coefs)

    def f_psi(self, xi: float | Array) -> Array:
        """Polynomial basis functions.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates.

        Returns
        -------
        array
            (`n_nodes`,) or (M, `n_nodes`,) array of polynomial basis
            functions.

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(self.f_psi, in_axes=0, out_axes=0)(xi)
        return self._f_psi(xi, self._psi_coefs)

    @staticmethod
    @jit
    def _f(xi: float, f_nodes: Array, psi_coefs: Array) -> float:
        """Value at coordinate from values at nodes.

        Parameters
        ----------
        xi : float
            Coordinate.
        f_nodes : array
            (`n_nodes`,) array of values at nodes.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions.

        Returns
        -------
        float
            Value at requested coordinate
            evaluated using polynomial basis functions.

        """
        psi = Basis._f_psi(xi, psi_coefs)
        return f_nodes @ psi

    @staticmethod
    @jit
    def _f_psi(xi: float, psi_coefs: Array) -> float:
        """Polynomial basis functions.

        Parameters
        ----------
        xi : float
            Coordinate.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions.

        Returns
        -------
        array
            (`n_nodes`,) array of polynomial basis functions.

        """
        psi = vmap(
            jnp.polyval,
            in_axes=(1, None),
            out_axes=0
        )(psi_coefs, xi)
        return psi

    @classmethod
    def Legendre(cls, n_nodes: int) -> Self:
        """Legendre basis functions.

        Parameters
        ----------
        n_nodes : int
            Number of nodes. Corresponds to the order of the basis
            functions.

        Returns
        -------
        basis
            Instance of the `Basis` class with nodes located at
            Gauss-Legendre roots.

        """
        xi, w = roots_legendre(n_nodes)
        xi = jnp.array(xi)
        w = jnp.array(w)
        return cls(xi, w=w)
