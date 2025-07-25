# -*- coding: utf-8 -*-
from typing import List
from typing_extensions import Self

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from ..basis import Basis
from ..borehole import Borehole
from ..path import Path


class Borefield:
    """Borefield.

    Parameters
    ----------
    boreholes : list of borehole
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

    def __init__(self, boreholes: List[Borehole]):
        self.boreholes = boreholes
        self.n_boreholes = len(boreholes)
        self.n_nodes = boreholes[0].n_nodes
        self.L = jnp.array([borehole.L for borehole in self.boreholes])

        # --- Basis functions ---
        self.f_psi = jit(
            lambda _eta: jnp.stack(
                [
                    vmap(borehole.f_psi, in_axes=0)(_eta) if len(jnp.shape(_eta)) > 0 else borehole.f_psi(_eta)
                    for borehole in boreholes
                ],
                axis=-2
            )
        )
        self.f = jit(
            lambda _eta, f_nodes: vmap(jnp.dot, in_axes=(-2, 0), out_axes=0)(self.f_psi(_eta), f_nodes)
        )

        # --- Nodal values of path and basis functions ---
        # Borehole coordinates (xi)
        self.xi = self.boreholes[0].xi
        # Positions (p)
        self.p = jnp.stack([borehole.p for borehole in self.boreholes], axis=0)
        # Derivatives of position (dp/dxi)
        self.dp_dxi = jnp.stack([borehole.dp_dxi for borehole in self.boreholes], axis=0)
        # Norms of the Jacobian (J)
        self.J = jnp.stack([borehole.J for borehole in self.boreholes], axis=0)
        # Longitudinal positions (s)
        self.s = jnp.stack([borehole.s for borehole in self.boreholes], axis=0)
        # Integration weights
        self.w = jnp.stack([borehole.w for borehole in self.boreholes], axis=0)

    def h_to_self(self, time: Array, alpha: float) -> Array:
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
            (K, `n_boreholes`, `n_nodes`, `n_boreholes`, `n_nodes`,) array
            of thermal response factors.
        """
        n_boreholes = self.n_boreholes
        h_to_self = jnp.zeros(
            (
                len(time),
                self.n_boreholes,
                self.n_nodes,
                self.n_boreholes,
                self.n_nodes
            )
        )
        for _i, borehole in enumerate(self.boreholes):
            # Thermal response factors onto nodes on itself
            h_to_self = h_to_self.at[:, _i, :, _i, :].set(
                borehole.h_to_self(time, alpha)
            )
            # Thermal response factors onto nodes of other boreholes
            # (up to borehole i - 1)
            p = self.p[:_i:, :, :].reshape(-1, 3)
            h_to_self = h_to_self.at[:, :_i, :, _i, :].set(
                borehole.h_to_point(p, time, alpha).reshape(len(time), -1, self.n_nodes, self.n_nodes)
            )
            # (borehole i + 1 and up)
            p = self.p[_i+1:, :, :].reshape(-1, 3)
            h_to_self = h_to_self.at[:, _i+1:, :, _i, :].set(
                borehole.h_to_point(p, time, alpha).reshape(len(time), -1, self.n_nodes, self.n_nodes)
            )
        return h_to_self

    def h_to_point(self, p: Array, time: Array, alpha: float, r_min: float = 0.) -> Array:
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
            (K, M, `n_boreholes`, `n_nodes`,) array of thermal response
            factors.
        """
        n_boreholes = self.n_boreholes
        h_to_point = jnp.stack(
            [
                self.boreholes[j].h_to_point(p, time, alpha, r_min=r_min) for j in jnp.arange(n_boreholes)
                ],
            axis=-2)
        return h_to_point

    @classmethod
    def from_dimensions(cls, L: ArrayLike, D: ArrayLike, r_b: ArrayLike, x: ArrayLike, y: ArrayLike, basis: Basis, n_segments: int, tilt: ArrayLike = 0., orientation: ArrayLike = 0., segment_ratios: ArrayLike | None = None, order: int = 101, order_to_self: int = 21) -> Self:
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
        borefield
            Instance of the `Borefield` class.

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
            path = Path.Line(L[j], D[j], x[j], y[j], tilt[j], orientation[j])
            boreholes.append(Borehole(r_b[j], path, basis, n_segments, segment_ratios=segment_ratios, order=order, order_to_self=order_to_self))
        return cls(boreholes)

    @classmethod
    def rectangle_field(cls, N_1: int, N_2: int, B_1: float, B_2: float, L: float, D: float, r_b: float, basis: Basis, n_segments: int, segment_ratios: ArrayLike | None = None, order: int = 101, order_to_self: int = 21) -> Self:
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
        basis : basis
            Basis functions.
        n_segments : int
            Number of segments per borehole.
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
        borefield
            Instance of the `Borefield` class.

        """
        # Borehole positions and orientation
        x = jnp.tile(jnp.arange(N_1), N_2) * B_1
        y = jnp.repeat(jnp.arange(N_2), N_1) * B_2
        return cls.from_dimensions(L, D, r_b, x, y, basis, n_segments, segment_ratios=segment_ratios, order=order, order_to_self=order_to_self)
        
