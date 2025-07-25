# -*- coding: utf-8 -*-
from functools import partial
from typing import Tuple
from typing_extensions import Self

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..basis import Basis
from ..heat_transfer import point_heat_source
from ..path import Path
from ..utilities import quad


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
    deg : int or None, default: None
        Polynomial degree at which to approximate the trajectory of
        the borehole. If None, `deg` is either set to `path.deg` (if
        it is not None) or to `n_nodes`. The longitudinal position along
        the borehole is approximated using a degree ``2 * `deg` - 1``.
    order : int, default: 101
        Order of the Gauss-Legendre quadrature to evaluate thermal
        response factors to points outside the borehole, and to evaluate
        coefficient matrices for fluid and heat extraction rate profiles.
    order_to_self : int, default: 21
        Order of the tanh-sinh quadrature to evaluate thermal
        response factors to nodes on the borehole. Corresponds to the
        number of quadrature points along each subinterval delimited
        by nodes and edges of the segments.

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

    def __init__(self, r_b: float, path: Path, basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None, deg: int | None = None, order: int = 101, order_to_self: int = 21):
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
        if deg is None:
            if path.deg is None:
                deg = basis.n_nodes
            else:
                deg = path.deg
        self.deg = deg
        self.order = order
        self.order_to_self = order_to_self
        # Segment edges
        xi_edges = 2. * jnp.cumulative_sum(segment_ratios, include_initial=True) - 1.
        self.xi_edges = xi_edges
        # Borehole length
        self.L = jnp.diff(path.f_s(jnp.array([-1., 1.])))[0]

        # --- Path functions ---
        self._initialize_path_coefficients(deg)

        # --- Nodal values of path and basis functions ---
        # Borehole coordinates (xi)
        xi = vmap(
            self.f_xi_sb,
            in_axes=(None, 0),
            out_axes=0
        )(basis.xi, jnp.arange(self.n_segments)).flatten()
        self.xi = xi
        # Positions (p)
        self.p = self.f_p(xi)
        # Derivatives of position (dp/dxi)
        self.dp_dxi = self.f_dp_dxi(xi)
        # Norms of the Jacobian (J)
        self.J = self.f_J(xi)
        # Longitudinal positions (s)
        self.s = self.f_s(xi)
        # Integration weights
        self.w = (jnp.tile(basis.w, (n_segments, 1)).T * segment_ratios).T.flatten() * self.J

    def f(self, xi: float | Array, f_nodes: Array) -> float | Array:
        """Value at coordinate from values at nodes.

        Parameters
        ----------
        xi : float or array
            Borehole coordinate or (M,) array of borehole coordinates.
        f_nodes : array
            (`n_nodes`,) array of values at borehole nodes.

        Returns
        -------
        float or array
            Value or (M,) array of values at requested coordinates
            evaluated using polynomial basis functions.

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(self.f, in_axes=(0, None), out_axes=0)(xi, f_nodes)
        xi_p, index = self.f_xi_bs(xi)
        f_nodes_segment = f_nodes.reshape(self.n_segments, -1)[index]
        return self.basis.f(xi_p, f_nodes_segment)

    def f_dp_dxi(self, xi: float | Array) -> Array:
        """Derivative of the position along the borehole.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        float
            (M,) or (M, 3,) array of derivatives of the position along the
            borehole (in meters).

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(
                self.f_dp_dxi,
                in_axes=0,
                out_axes=0
            )(xi)
        xi_p, index = self.f_xi_bs(xi)
        return self._derivative_of_position(xi_p, index, self._dp_dxi_coefs) / self.segment_ratios[index]

    def f_J(self, xi: float | Array) -> float | Array:
        """Norm of the Jacobian along the borehole.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        float
            Norm of the Jacobian or (M,) array of norms of the Jacobian
            (in meters).

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(
                self.f_J,
                in_axes=0,
                out_axes=0
            )(xi)
        xi_p, index = self.f_xi_bs(xi)
        return self._norm_of_jacobian(xi_p, index, self._J_coefs) / self.segment_ratios[index]

    def f_p(self, xi: float | Array) -> Array:
        """Position along the borehole.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        float
            (3,) or (M, 3,) array of positions along the borehole
            (in meters).

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(
                self.f_p,
                in_axes=0,
                out_axes=0
            )(xi)
        xi_p, index = self.f_xi_bs(xi)
        return self._position(xi_p, index, self._p_coefs)

    def f_s(self, xi: float | Array) -> float | Array:
        """Longitudinal position along the borehole.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        float
            Longitudinal position or (M,) array of longitudinal
            positions along the borehole (in meters).

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(
                self.f_s,
                in_axes=0,
                out_axes=0
            )(xi)
        xi_p, index = self.f_xi_bs(xi)
        return self._longitudinal_position(xi_p, index, self._s_coefs)

    def f_xi_bs(self, xi: float | Array) -> Tuple[float | Array, int]:
        """Segment coordinate from borehole coordinate.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates along the borehole.

        Returns
        -------
        xi_p : float or array
            Coordinate or (M,) array of coordinates along the borehole
            segments.
        index : int or array
            Index or (M,) array of indices of the borehole segments.

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(
                self.f_xi_bs,
                in_axes=0,
                out_axes=0
            )(xi)
        return self._segment_coordinate(xi, self.xi_edges)

    def f_xi_sb(self, xi_p: float, index: int) -> float:
        """Borehole coordinate from segment coordinate.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.

        Returns
        -------
        xi : float
            Coordinate along the borehole.

        """
        return self._borehole_coordinate(xi_p, self.xi_edges, index)

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
        n_nodes = self.n_nodes
        if len(jnp.shape(time)) > 0:
            n_times = len(time)
            if len(jnp.shape(p)) > 1:
                shape = (n_times, -1, n_nodes)
            else:
                shape = (n_times, n_nodes)
            h_to_point = vmap(
                self.h_to_point,
                in_axes=(None, 0, None, None)
            )(
                p, time, alpha, r_min
            ).reshape(shape)
            return h_to_point
        if len(jnp.shape(p)) > 1:
            h_to_point = vmap(
                self.h_to_point,
                in_axes=(0, None, None, None)
            )(
                p, time, alpha, r_min
            ).reshape(-1, n_nodes)
            return h_to_point
        h_to_point = self._h_to_point(p, time, alpha, r_min)
        return h_to_point

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
        n_nodes = self.n_nodes
        if len(jnp.shape(time)) > 0:
            n_times = len(time)
            h_to_self = vmap(
                self.h_to_self,
                in_axes=(0, None)
            )(
                time, alpha
            )
            return h_to_self
        h_to_self = vmap(
            self._h_to_node_on_self,
            in_axes=(0, None, None)
        )(
            self.p, time, alpha
        ).reshape(n_nodes, n_nodes)
        return h_to_self

    def _h_to_node_on_self(self, p: Array, time: float, alpha: float):
        """Thermal response factors to its own nodes.

        Parameters
        ----------
        p : array
            (3,) array of the position of a node on the borehole at
            which thermal response factors are evaluated.
        time : float
            Time (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).

        Returns
        -------
        array
            (`n_nodes`,) array of thermal response factors.
        """
        h_to_node = vmap(
            self._thermal_response_factor_to_self,
            in_axes=(0, None, None, None, None, None, None, None, None, None),
            out_axes=0
        )(
            jnp.arange(self.n_segments),
            p,
            time,
            alpha,
            self.r_b,
            self._p_coefs,
            self._J_coefs,
            self.basis._psi_coefs,
            self.basis.xi,
            self.order_to_self
        )
        return h_to_node.flatten()

    def _h_to_point(self, p: Array, time: float, alpha: float, r_min: float = 0.):
        """Thermal response factors to a point.

        Parameters
        ----------
        p : array
            (3,) array of the position at which thermal response factors
            are evaluated.
        time : float
            Time (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).
        r_min : float, default: ``0.``
            Minimum distance (in meters) added to the distance between the
            heat source and the positions `p`.

        Returns
        -------
        array
            (`n_nodes`,) array of thermal response factors.
        """
        h_to_node = vmap(
            self._thermal_response_factor,
            in_axes=(0, None, None, None, None, None, None, None, None),
            out_axes=0
        )(
            jnp.arange(self.n_segments),
            p,
            time,
            alpha,
            r_min,
            self._p_coefs,
            self._J_coefs,
            self.basis._psi_coefs,
            self.order
        )
        return h_to_node.flatten()

    def _initialize_path_coefficients(self, deg: int):
        """Initialize path coefficients.

        Parameters
        ----------
        deg : int
            Polynomial degree to approximate the path along each
            segment. The longitudinal position ``s`` will be approximated
            using degree ``2 * deg - 1`` polynomials.

        """
        xi_p = jnp.linspace(-1, 1, num=deg+1)
        xi = vmap(
            self.f_xi_sb, 
            in_axes=(None, 0),
            out_axes=1
        )(xi_p, jnp.arange(self.n_segments))
        p = vmap(
            self.path.f_p,
            in_axes=1,
            out_axes=1
        )(xi)
        self._p_coefs = vmap(
            vmap(
                jnp.polyfit,
                in_axes=(None, 1, None),
                out_axes=1),
            in_axes=(None, 2, None),
            out_axes=2
        )(xi_p, p, deg)
        self._dp_dxi_coefs = vmap(
            vmap(
                jnp.polyder,
                in_axes=1,
                out_axes=1),
            in_axes=2,
            out_axes=2
        )(self._p_coefs)
        xi_p = jnp.linspace(-1, 1, num=2*deg)
        xi = vmap(
            self.f_xi_sb, 
            in_axes=(None, 0),
            out_axes=1
        )(xi_p, jnp.arange(self.n_segments))
        s = vmap(
            self.path.f_s,
            in_axes=1,
            out_axes=1
        )(xi)
        self._s_coefs = vmap(
            jnp.polyfit,
            in_axes=(None, 1, None),
            out_axes=1
        )(xi_p, s, 2*deg-1)
        self._J_coefs = vmap(
            jnp.polyder,
            in_axes=1,
            out_axes=1
        )(self._s_coefs)

    @staticmethod
    @jit
    def _borehole_coordinate(xi_p: float, xi_edges: Array, index: int) -> float:
        """Borehole coordinate from segment coordinate.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        xi_edges : array
            (`n_segments`+1,) array of the coordinates of the edges of
            the segments.
        index : int
            Index of the borehole segment.

        Returns
        -------
        xi : float
            Coordinate along the borehole.

        """
        a, b = xi_edges[index], xi_edges[index + 1]
        xi = 0.5 * (b + a) + 0.5 * xi_p * (b - a)
        return xi

    @staticmethod
    @jit
    def _derivative_of_position(xi_p: float, index: int, dp_dxi_coefs: Array) -> float:
        """Derivative of the position along a borehole segment.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        dp_dxi_coefs : array
            (`n_nodes`, `n_segments`, 3) array of polynomial coefficients
            for the evaluation of the derivatives of the position (in meters)
            along the borehole segments as a function of `xi_p`.

        Returns
        -------
        float
            (3,) array of the derivative of the position along the borehole
            segment (in meters).

        """
        dp_dxi = vmap(
            jnp.polyval,
            in_axes=(1, None),
            out_axes=0
        )(dp_dxi_coefs[:, index, :], xi_p)
        return dp_dxi

    @staticmethod
    @jit
    def _longitudinal_position(xi_p: float, index: int, s_coefs: Array) -> float:
        """Longitudinal position along a borehole segment.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        float
            Longitudinal position along the borehole segment (in meters).

        """
        return jnp.polyval(s_coefs[:, index], xi_p)

    @staticmethod
    @jit
    def _norm_of_jacobian(xi_p: float, index: int, J_coefs: Array) -> float:
        """Norm of the Jacobian along a borehole segment.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        J_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the norm of the jacobian (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        float
            Norm of the Jacobian along the borehole segment (in meters).

        """
        return jnp.polyval(J_coefs[:, index], xi_p)

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _point_heat_source(cls, xi_p: float, index: int, p: Array, time: float, alpha: float, r_min: float, p_coefs: Array) -> Array:
        """Point heat source solution.

        Parameters
        ----------
        xi_p : array
            (N,) array of coordinates along segments.
        p : array
            (3,) array of positions.
        time : float
            Time (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).
        r_min : float
            Minimum distance (in meters) added to the distance between the
            heat source and the positions `p`.
        p_coefs : array
            (`n_nodes`, `n_segments`, 3) array of polynomial coefficients
            for the evaluation of the position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        array
            (N,) array of the point heat source solution.
        """
        p_source = cls._position(xi_p, index, p_coefs)
        p_mirror = p * jnp.array([1, 1, -1], dtype=int)
        # Point heat source solutions
        h = (
            point_heat_source(p_source, p, time, alpha, r_min)
            - point_heat_source(p_source, p_mirror, time, alpha, r_min)
        )
        return h

    @staticmethod
    @jit
    def _position(xi_p: float, index: int, p_coefs: Array) -> float:
        """Longitudinal position along a borehole segment.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        p_coefs : array
            (`n_nodes`, `n_segments`, 3) array of polynomial coefficients
            for the evaluation of the position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        float
            (3,) array of the position along the borehole segment (in meters).

        """
        p = vmap(
            jnp.polyval,
            in_axes=(1, None),
            out_axes=0
        )(p_coefs[:, index, :], xi_p)
        return p

    @staticmethod
    @partial(jit, static_argnames=['index_out'])
    def _segment_coordinate(xi: float, xi_edges: Array, index_out: bool = True) -> Tuple[float, int] | float:
        """Segment coordinate from borehole coordinate.

        Parameters
        ----------
        xi : float
            Coordinate along the borehole.
        xi_edges : array
            (`n_segments` + 1,) array of coordinates of the edges of
            the segments.
        index_out : bool, default: True
            Set to True to return the index of the borehole segment.

        Returns
        -------
        xi_p : float
            Coordinate along the borehole segment.
        index : int, optional
            Index of the borehole segment. Returned only if `index_out`
            is True.

        """
        index = jnp.maximum(
            jnp.searchsorted(xi_edges, xi),
            1
        ).astype(int) - 1
        a, b = xi_edges[index], xi_edges[index + 1]
        xi_p = (2 * xi - (b + a)) / (b - a)
        if index_out:
            return xi_p, index
        else:
            return xi_p

    @classmethod
    @partial(jit, static_argnames=['cls', 'order'])
    def _thermal_response_factor(cls, index: int, p: Array, time: float, alpha: float, r_min: float, p_coefs: Array, J_coefs: Array, psi_coefs: Array, order: int = 101) -> Array:
        """Point heat source solution.

        Parameters
        ----------
        index : int
            Index of the borehole segment.
        p : array
            (3,) array of positions.
        time : float
            Time (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).
        r_min : float
            Minimum distance (in meters) added to the distance between the
            heat source and the positions `p`.
        p_coefs : array
            (`n_nodes`, `n_segments`, 3) array of polynomial coefficients
            for the evaluation of the position (in meters) along
            the borehole segments as a function of `xi_p`.
        J_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the norm of the jacobian (in meters) along
            the borehole segments as a function of `xi_p`.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions along
            the borehole segments as a function of `xi_p`.
        order : int, default: 101
            Order of the numerical quadrature.

        Returns
        -------
        array
            (`n_nodes`,) array of the point heat source solution.
        """
        def thermal_reponse_factor_integrand(xi_p: float) -> Array:
            """Integrand of the thermal response factor to a point."""
            J = cls._norm_of_jacobian(xi_p, index, J_coefs)
            psi = Basis._f_psi(xi_p, psi_coefs)
            h_point_source = psi * J * cls._point_heat_source(
                xi_p, index, p, time, alpha, r_min, p_coefs)
            return h_point_source

        h = quad(
            thermal_reponse_factor_integrand,
            order=order,
            rule='gl'
        )
        return h

    @classmethod
    @partial(jit, static_argnames=['cls', 'order'])
    def _thermal_response_factor_to_self(cls, index: int, p: Array, time: float, alpha: float, r_min: float, p_coefs: Array, J_coefs: Array, psi_coefs: Array, xi: Array, order: int = 21) -> Array:
        """Point heat source solution.

        Parameters
        ----------
        index : int
            Index of the borehole segment.
        p : array
            (3,) array of positions.
        time : float
            Time (in seconds).
        alpha : float
            Ground thermal diffusivity (in m^2/s).
        r_min : float
            Minimum distance (in meters) added to the distance between the
            heat source and the positions `p`.
        p_coefs : array
            (`n_nodes`, `n_segments`, 3) array of polynomial coefficients
            for the evaluation of the position (in meters) along
            the borehole segments as a function of `xi_p`.
        J_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the norm of the jacobian (in meters) along
            the borehole segments as a function of `xi_p`.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions along
            the borehole segments as a function of `xi_p`.
        order : int, default: 21
            Order of the numerical quadrature along each subinterval
            delimited by segment edges and nodes.

        Returns
        -------
        array
            (`n_nodes`,) array of the point heat source solution.
        """
        def thermal_reponse_factor_to_self_integrand(xi_p: float) -> Array:
            """Integrand of the thermal response factor to a node."""
            J = cls._norm_of_jacobian(xi_p, index, J_coefs)
            psi = Basis._f_psi(xi_p, psi_coefs)
            h_point_source = psi * J * cls._point_heat_source(
                xi_p, index, p, time, alpha, r_min, p_coefs)
            return h_point_source

        points = jnp.concatenate(
            [
                -jnp.ones(1),
                xi,
                jnp.ones(1)
            ]
        )

        h = quad(
            thermal_reponse_factor_to_self_integrand,
            points=points,
            order=order,
            rule='ts'
        )
        return h

    @classmethod
    def from_dimensions(cls, L: float, D: float, r_b: float, x: float, y: float, basis: Basis, n_segments: int, tilt: float = 0., orientation: float = 0., segment_ratios: ArrayLike | None = None, order: int = 101, order_to_self: int = 21) -> Self:
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
        borehole
            Instance of the `Borehole` class.

        """
        path = Path.Line(L, D, x, y, tilt, orientation)
        return cls(r_b, path, basis, n_segments, segment_ratios=segment_ratios, order=order, order_to_self=order_to_self)
