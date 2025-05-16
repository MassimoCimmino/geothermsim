# -*- coding: utf-8 -*-
from functools import partial
from typing import Self

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..basis import Basis
from ..path import Path


class Borehole:
    """Geothermal borehole.

    Parameters
    ----------
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

    Attributes
    ----------
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

    def __init__(self, r_b: float, path: Path, basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None):
        # Runtime type validation
        if not isinstance(segment_ratios, ArrayLike) and segment_ratios is not None:
            raise TypeError(f"Expected arraylike or None input; got {segment_ratios}")
        # Convert input to jax.Array
        if segment_ratios is None:
            segment_ratios = jnp.full(n_segments, 1. / n_segments)
        else:
            segment_ratios = jnp.asarray(segment_ratios)

        # --- Class attributes ---
        self.r_b = r_b
        self.path = path
        self.basis = basis
        self.n_segments = n_segments
        self.n_nodes = basis.n_nodes * n_segments
        self.segment_ratios = segment_ratios
        # Segment edges
        xi_edges = 2. * jnp.cumulative_sum(segment_ratios, include_initial=True) - 1.
        self.xi_edges = xi_edges
        # Borehole length
        self.L = jnp.diff(path.f_s(jnp.array([-1., 1.])))[0]

        # --- Changes of coordinates ---
        a, b = xi_edges[:-1], xi_edges[1:]
        # Segments --> Borehole
        f_xi_sb = lambda _eta: 0.5 * (b + a) + 0.5 * _eta * (b - a)
        self.f_xi_sb = jit(
            lambda _eta: vmap(f_xi_sb, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else f_xi_sb(_eta)
        )
        # Borehole --> Segments
        f_xi_bs = lambda _eta: (2 * _eta - (b + a)) / (b - a)
        self.f_xi_bs = jit(
            lambda _eta: vmap(f_xi_bs, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else f_xi_bs(_eta)
        )

        # --- Basis functions ---
        # Weights at interfaces between segments
        w_interfaces = jnp.concatenate([jnp.array([1.]), jnp.full(n_segments - 1, 0.5), jnp.array([1.])])
        f_w_interfaces = lambda _eta: jnp.heaviside(b - _eta, w_interfaces[1:]) * jnp.heaviside(_eta - a, w_interfaces[:-1])
        f_psi = lambda _eta: vmap(
            lambda _eta_p, _in_segment: basis.f_psi(_eta_p) * _in_segment,
            in_axes=(0, 0)
        )(f_xi_bs(_eta), f_w_interfaces(_eta)).flatten()
        self.f_psi = jit(
            lambda _eta: vmap(f_psi, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else f_psi(_eta)
        )
        self.f = jit(
            lambda _eta, f_nodes: self.f_psi(_eta) @ f_nodes
        )

        # --- Nodal values of path and basis functions ---
        # Borehole coordinates (xi)
        xi = self.f_xi_sb(basis.xi).T.flatten()
        self.xi = xi
        # Positions (p)
        self.p = path.f_p(xi)
        # Derivatives of position (dp/dxi)
        self.dp_dxi = path.f_dp_dxi(xi)
        # Norms of the Jacobian (J)
        self.J = path.f_J(xi)
        # Longitudinal positions (s)
        self.s = path.f_s(xi)
        # Integration weights
        self.w = (jnp.tile(basis.w, (n_segments, 1)).T * segment_ratios).T.flatten() * self.J

    def h_to_borehole(self, borehole: Self, time: ArrayLike, alpha: float) -> Array:
        """Thermal response factors to nodes of another borehole.

        Parameters
        ----------
        borehole : borehole
            Borehole for which the thermal response factors will be
            evaluated at the nodes.
        time : array
            (K,) array of times (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).

        Returns
        -------
        array
            (K, ```borehole`.n_nodes``, `n_nodes`,) array of thermal
            response factors.
        """
        return self.h_to_point(borehole.p, time, alpha)

    @partial(jit, static_argnames=['self'])
    def h_to_coordinate_on_self(self, xi: Array, time: Array, alpha: float) -> Array:
        """Thermal response factors to coordinates along itself.

        Parameters
        ----------
        xi : array
            (M,) array of the coordinates along itself at which thermal
            response factors are evaluated.
        time : array
            (K,) array of times (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).

        Returns
        -------
        array
            (K, M, `n_nodes`,) array of thermal response factors.
        """
        # Positions (p) of points on self
        p = self.path.f_p(xi)
        return self.h_to_point(p, time, alpha, r_min=self.r_b)

    @partial(jit, static_argnames=['self'])
    def h_to_point(self, p: Array, time: Array, alpha: float, r_min: float = 0.):
        """Thermal response factors to a point.

        Parameters
        ----------
        p : array
            (M, 3,) array of the positions at which thermal response
            factors are evaluated.
        time : array
            (K,) array of times (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).
        r_min : float, default: ``0.``
            Minimum distance (in meters) added to the distance between the
            heat source and the positions `p`.

        Returns
        -------
        array
            (K, M, `n_nodes`,) array of thermal response factors.
        """
        # Integrand of point heat source
        integrand = vmap(
            lambda _eta: self._segment_point_heat_source(_eta, p, time, alpha, r_min=r_min),
            in_axes=0,
            out_axes=-1)
        n_times = len(time)
        n_nodes = self.n_nodes
        h_to_point = self.basis.quad_gl(
            integrand, -1., 1.
        ).reshape(n_times, -1, n_nodes)
        return h_to_point

    @partial(jit, static_argnames=['self'])
    def h_to_self(self, time: Array, alpha: float):
        """Thermal response factors to its own nodes.

        Parameters
        ----------
        time : array
            (K,) array of times (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).

        Returns
        -------
        array
            (K, `n_nodes`, `n_nodes`,) array of thermal response factors.
        """
        # Integrand of point heat source evaluated at borehole nodes
        integrand = vmap(
            lambda _eta: self._segment_point_heat_source(_eta, self.p, time, alpha, r_min=self.r_b),
            in_axes=0,
            out_axes=-1)
        # Integral of the point heat source
        n_times = len(time)
        n_nodes = self.n_nodes
        h_to_self = self.basis.quad_ts_nodes(
            integrand
        ).reshape(n_times, n_nodes, n_nodes)
        return h_to_self

    @partial(jit, static_argnames=['self'])
    def _segment_point_heat_source(self, xi_p: float, p: Array, time: Array | float, alpha: float, r_min: float = 0.) -> Array:
        """Point heat source solution along all segments of the borehole.

        Parameters
        ----------
        xi_p : array
            (N,) array of coordinates along segments.
        p : array
            (M, 3,) array of positions.
        time : array or float
            (K,) array of times (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).
        r_min : float, default: ``0.``
            Minimum distance (in meters) added to the distance between the
            heat source and the positions `p`.

        Returns
        -------
        array
            (K, M, N,) array of the point heat source solution. If `time`
            is a ``float``, then ``K=0``.
        """
        # Coordinates (xi) of all sources at local segment coordinates (xi')
        xi = self.f_xi_sb(xi_p)
        # Point heat source solutions
        h = self.path.point_heat_source(xi, p, time, alpha, r_min=r_min) * self.segment_ratios
        return h

    @classmethod
    def from_dimensions(cls, L: float, D: float, r_b: float, x: float, y: float, basis: Basis, n_segments: int, tilt: float = 0., orientation: float = 0., segment_ratios: ArrayLike | None = None) -> Self:
        """Straight borehole from its dimensions.

        Parameters
        ----------
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

        Returns
        -------
        borehole
            Instance of the `Borehole` class.

        """
        path = Path.from_dimensions(L, D, x, y, tilt, orientation)
        return cls(r_b, path, basis, n_segments, segment_ratios=segment_ratios)
