# -*- coding: utf-8 -*-
from collections.abc import Callable
from functools import partial
from typing_extensions import Self

from interpax import Interpolator1D
from jax import numpy as jnp
from jax import Array, jacobian, jit, vmap
from jax.typing import ArrayLike
from jax.scipy.special import erfc
from scipy.integrate import fixed_quad


class Path:
    """Trajectory of a borehole.

    Parameters
    ----------
    f_p : callable
        Function that takes a coordinate or an (N,) array of
        coordinates ``xi`` and returns an (3,) or (N, 3,) array of
        positions [x, y, z] along the path.
    f_dp_dxi : callable
        Function that takes a coordinate or an (N,) array of
        coordinates ``xi`` and returns an (3,) or (N, 3,) array of
        derivatives [dx/dxi, dy/dxi, dz/dxi] along the path.
    f_J : callable
        Function that takes a coordinate or an (N,) array of
        coordinates ``xi`` and returns a float or an (N,) array of
        norms of the Jacobian sqrt(dx/dxi**2 + dy/dxi**2 + dz/dxi**2)
        along the path.
    f_s : callable
        Function that takes a coordinate or an (N,) array of
        coordinates ``xi`` and returns a float or an (N,) array of
        the longitudinal position ``s`` (in meters) along the path.
    xi : array_like or None, default: None
        (`n_nodes`,) array of node coordinates along the interval
        ``[-1, 1]``.
    p : array_like or None, default: None
        Positions (``x``, ``y``, ``z``) of the nodes along the
        trajectory of the borehole.
    deg : int or None, default: None
        Polynomial degree of the path. This is used as a default to
        approximate the trajectory in borehole models.

    Attributes
    ----------
    n_nodes : int
        Number of nodes.
    L : float
        Length of the path (in meters)

    """

    def __init__(self, f_p: Callable[[float | Array], Array], f_dp_dxi: Callable[[float | Array], Array], f_J: Callable[[float | Array], float | Array], f_s: Callable[[float | Array], float | Array], xi: ArrayLike | None = None, p: ArrayLike | None = None, deg: int | None = None):
        # Runtime type validation
        if not isinstance(xi, ArrayLike) and xi is not None:
            raise TypeError(f"Expected arraylike input; got {xi}")
        if not isinstance(p, ArrayLike) and p is not None:
            raise TypeError(f"Expected arraylike input; got {p}")
        # Convert input to jax.Array
        xi = jnp.asarray(xi)
        p = jnp.atleast_2d(p)

        # --- Class atributes ---
        self.xi = xi
        self.p = p
        self.deg = deg

        # --- Path functions ---
        # Position (p)
        self._f_p = f_p
        # Derivative of position (dp/dxi)
        self._f_dp_dxi = f_dp_dxi
        # Norm of the Jacobian (J)
        self._f_J = f_J
        # Longitudinal position (s)
        self._f_s = f_s

        # --- Additional attributes ---
        # Number of nodes
        if xi is not None and p is not None:
            n_nodes = len(xi)
            self.n_nodes = n_nodes
        else:
            self.n_nodes = None
        # Length of the path
        self.L = self.f_s(1.)

    def f_p(self, xi: float | Array) -> Array:
        """Position along the path.

        Parameters
        ----------
        xi : float or array
            Coordinate of (N,) array of coordinates along the
            trajectory.

        Returns
        -------
        array
            (3,) or (N, 3,) array of positions
            (``x``, ``y``, ``z``) along the trajectory.
        """
        return self._f_p(xi)

    def f_dp_dxi(self, xi: float | Array) -> Array:
        """Derivative of the position along the path.

        Parameters
        ----------
        xi : float or array
            Coordinate of (N,) array of coordinates along the
            trajectory.

        Returns
        -------
        array
            (3,) or (N, 3,) array of derivatives of the position
            (``dx/dxi``, ``dy/dxi``, ``dz/xi``) along the trajectory.
        """
        return self._f_dp_dxi(xi)

    def f_J(self, xi: float | Array) -> float | Array:
        """Norm of the Jacobian along the path.

        Parameters
        ----------
        xi : float or array
            Coordinate of (N,) array of coordinates along the
            trajectory.

        Returns
        -------
        float or array
            Norm of the Jacobian or (N,) array of the norms of the
            Jacobian along the trajectory.
        """
        return self._f_J(xi)

    def f_s(self, xi: float | Array) -> float | Array:
        """Longitudinal position along the path.

        Parameters
        ----------
        xi : float or array
            Coordinate of (N,) array of coordinates along the
            trajectory.

        Returns
        -------
        float or array
            Longitudinal position or (N,) array of longitudinal
            positions along the trajectory.
        """
        return self._f_s(xi)

    @classmethod
    def Line(cls, L: float, D: float, x: float, y: float, tilt: float, orientation: float) -> Self:
        """Path from the dimensions of a borehole.

        Parameters
        ----------
        L : float
            Borehole length (in meters).
        D : float
            Borehole buried depth (in meters).
        r_b : float
            Borehole radius (in meters).
        x, y : array_like
            Horizontal position (in meters) of the top end of the
            borehole.
        tilt : float
            Tilt angle (in radians) of the borehole with respect to
            vertical.
        orientation : float
            Orientation (in radians) of the inclined borehole. An
            inclination toward the x-axis corresponds to an orientation
            of zero.

        Returns
        -------
        path
            Instance of the `Path` class.

        """
        # Position of the top of the borehole
        p0 = jnp.array([x, y, -D])
        # Position of the bottom of the borehole
        delta_p = jnp.array(
            [
                L * jnp.sin(tilt) * jnp.cos(orientation),
                L * jnp.sin(tilt) * jnp.sin(orientation),
                -L * jnp.cos(tilt)
            ]
        )
        p1 = p0 + delta_p

        # --- Class atributes ---
        xi = jnp.array([-1., 1.])
        p = jnp.stack([p0, p1], axis=0)

        # --- Path functions ---
        p_mean = p.mean(axis=0)
        L = jnp.linalg.norm(delta_p)
        def f_p(_xi: float | Array) -> Array:
            """Position along the path."""
            if len(jnp.shape(_xi)) > 0:
                return vmap(f_p, in_axes=0)(_xi)
            return p_mean + 0.5 * _xi * delta_p
        def f_dp_dxi(_xi: float | Array) -> Array:
            """Derivative of the position along the path."""
            if len(jnp.shape(_xi)) > 0:
                return jnp.broadcast_to(0.5 * delta_p, (len(_xi), 3))
            return 0.5 * delta_p
        def f_J(_xi: float | Array) -> float | Array:
            """Norm of the Jacobian along the path."""
            if len(jnp.shape(_xi)) > 0:
                return jnp.broadcast_to(0.5 * L, len(_xi))
            return 0.5 * L
        def f_s(_xi: float | Array) -> float | Array:
            """Longitudinal position along the path."""
            return 0.5 * (1 + _xi) * L
        return cls(f_p, f_dp_dxi, f_J, f_s, xi=xi, p=p, deg=1)

    @classmethod
    def Polynomial(cls, xi: ArrayLike, p: ArrayLike, deg: int = None, s_method: str = 'monotonic', s_order: int = 21, s_num: int = 21) -> Self:
        """Polynomial path from positions along the path.

        Parameters
        ----------
        xi : array_like
            (N,) array of coordinates along the path. Should be of
            sufficient length for the selected interpolation methods.
        p : array_like
            (N, 3,) array of positions (in meters) along the path.
        deg : int or None, default: None
            Degree of the polynomial representing the path. If None,
            ``deg = N - 1``. If ``deg < N - 1``, the polynomial is
            obtained by regression, taking the first position as fixed
            (usually ``xi[0] = -1.`` and its corresponding position
            ``p[0]``).
        s_method : str, default: 'monotonic'
            Interpolation methods to be used by
            ``interpax.Interpolator1D`` for the position along the
            path and for the longitudinal position ``s`` along the
            path.
        s_order : int, default: 21
            Number of points for the integration of the Jacobian between
            each subsequent coordinates to obtain the ongitudinal
            position ``s`` from the norm of the Jacobian along the path.
        s_num : int, default: 21
            Number of evenly distributed knots along the path to
            evaluate the longitudinal position ``s`` using trapezoidal
            integration and generate the interpolator.
            

        Returns
        -------
        path
            Instance of the `Path` class.

        """
        # Runtime type validation
        if not isinstance(xi, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {xi}")
        if not isinstance(p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {p}")
        # Convert input to jax.Array
        xi = jnp.asarray(xi)
        p = jnp.atleast_2d(p)

        # --- Path functions ---
        # Position along the path
        if deg is None:
            p_coefs = jnp.polyfit(xi, p, len(xi)-1)
            deg = len(xi)-1
        else:
            p_w = jnp.ones_like(xi)
            p_w.at[0].set(1e4)
            p_coefs = jnp.polyfit(xi, p, deg, w=p_w)
        def f_p(_xi: float | Array) -> Array:
            """Position along the path."""
            if len(jnp.shape(_xi)) > 0:
                return vmap(f_p, in_axes=0)(_xi)
            return jnp.polyval(p_coefs, _xi)
        # Derivative of the position along the path
        dp_coefs = vmap(
            jnp.polyder,
            in_axes=-1,
            out_axes=-1
        )(p_coefs)
        def f_dp_dxi(_xi: float | Array) -> Array:
            """Derivative of the position along the path."""
            if len(jnp.shape(_xi)) > 0:
                return vmap(f_dp_dxi, in_axes=0)(_xi)
            return jnp.polyval(dp_coefs, _xi)
        # Norm of the Jacobian along the path
        dp_square_coefs = vmap(
            jnp.polymul,
            in_axes=-1,
            out_axes=-1
        )(dp_coefs, dp_coefs)
        def f_J(_xi: float | Array) -> float | Array:
            """Norm of the Jacobian along the path."""
            if len(jnp.shape(_xi)) > 0:
                return vmap(f_J, in_axes=0)(_xi)
            return jnp.sqrt(jnp.polyval(dp_square_coefs, _xi).sum(axis=-1))
        # Longitudinal position along the path
        s_xi = jnp.linspace(-1., 1., num=s_num)
        a, b = s_xi[:-1], s_xi[1:]
        ds = jnp.array(
            [
                fixed_quad(f_J, _a, _b, n=s_order)[0]
                for _a, _b in zip(a, b)
            ]
        )
        s = jnp.cumulative_sum(ds, include_initial=True)
        f_s = Interpolator1D(s_xi, s, method=s_method, extrap=True)
        return cls(f_p, f_dp_dxi, f_J, f_s, xi=xi, p=p, deg=deg)

    @classmethod
    def Spline(cls, xi: ArrayLike, p: ArrayLike, method: str = 'cubic2', s_method: str = 'monotonic', s_order: int = 21, s_num: int = 21) -> Self:
        """Path from positions along the path.

        Parameters
        ----------
        xi : array_like
            (N,) array of coordinates along the path. Should be of
            sufficient length for the selected interpolation methods.
        p : array_like
            (N, 3,) array of positions (in meters) along the path.
        method, s_method : str, default: 'cubic2', 'monotonic'
            Interpolation methods to be used by
            ``interpax.Interpolator1D`` for the position along the
            path and for the longitudinal position ``s`` along the
            path.
        s_order : int, default: 21
            Number of points for the integration of the Jacobian between
            each subsequent coordinates to obtain the ongitudinal
            position ``s`` from the norm of the Jacobian along the path.
        s_num : int, default: 21
            Number of evenly distributed knots along the path to
            evaluate the longitudinal position ``s`` using trapezoidal
            integration and generate the interpolator.
            

        Returns
        -------
        path
            Instance of the `Path` class.

        """
        # Runtime type validation
        if not isinstance(xi, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {xi}")
        if not isinstance(p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {p}")
        # Convert input to jax.Array
        xi = jnp.asarray(xi)
        p = jnp.atleast_2d(p)

        # --- Path functions ---
        # Position along the path
        f_p = Interpolator1D(xi, p, method=method, extrap=True)
        # Derivative of the position along the path
        f_dp_dxi = partial(f_p, dx=1)
        # Norm of the Jacobian along the path
        def f_J(_xi: float | Array) -> float | Array:
            """Norm of the Jacobian along the path."""
            return jnp.linalg.norm(f_dp_dxi(_xi), axis=-1)
        # Longitudinal position along the path
        s_xi = jnp.linspace(-1., 1., num=s_num)
        a, b = s_xi[:-1], s_xi[1:]
        ds = jnp.array(
            [
                fixed_quad(f_J, _a, _b, n=s_order)[0]
                for _a, _b in zip(a, b)
            ]
        )
        s = jnp.cumulative_sum(ds, include_initial=True)
        f_s = Interpolator1D(s_xi, s, method=s_method, extrap=True)
        return cls(f_p, f_dp_dxi, f_J, f_s, xi=xi, p=p, deg=None)
