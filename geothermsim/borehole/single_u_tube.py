# -*- coding: utf-8 -*-
from functools import partial
from typing import Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import fori_loop, switch

from ._tube import _Tube
from ..basis import Basis
from ..utilities import quadgl


class SingleUTube(_Tube):
    """Single U-Tube geothermal borehole.

    Parameters
    ----------
    R_d : array_like or callable
        (2, 2,) array of thermal resistances (in m-K/W), or callable that
        takes the mass flow rate as input (in kg/s) and returns a (2, 2,)
        array.
    fluid : fluid
        The fluid.
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

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _fluid_temperature_a_in(cls, xi_p: float, index: int, beta_ij: Array, top_connectivity: Tuple[Array, Array], s_coefs: Array) -> Array:
        """Inlet coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij : array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole as a function of `xi_p`.

        Returns
        -------
        array
            (`n_pipes`,) array of coefficients for the inlet fluid
            temperature.

        """
        b_in = cls._outlet_fluid_temperature_a_in(
            beta_ij, top_connectivity, None, s_coefs)
        c_in = cls._general_solution_a_in(
            xi_p, index, beta_ij, s_coefs)
        c_out = cls._general_solution_a_out(
            xi_p, index, beta_ij, s_coefs)
        a_in = c_in + b_in * c_out
        return a_in

    @classmethod
    @partial(jit, static_argnames=['cls', 'order'])
    def _fluid_temperature_a_b(cls, xi_p: float, index: int, beta_ij: Array, top_connectivity: Tuple[Array, Array], s_coefs: Array, J_coefs: Array, psi_coefs: Array, order: int = 101) -> Array:
        """Borehole wall coefficient to evaluate the fluid temperatures.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij : array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
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
            (`n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        b_b = cls._outlet_fluid_temperature_a_b(
            beta_ij, top_connectivity, None, s_coefs, J_coefs, psi_coefs, order)
        c_out = cls._general_solution_a_out(
            xi_p, index, beta_ij, s_coefs)
        c_b = cls._general_solution_a_b(
            xi_p, index, beta_ij, s_coefs, J_coefs, psi_coefs, order)
        a_b = c_b + jnp.outer(c_out, b_b)
        return a_b

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _general_solution_a_in(cls, xi_p: float, index: int, beta_ij: Array, s_coefs: Array) -> Array:
        """Inlet coefficient to evaluate the general solution.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        array
            (2,) array of coefficients for the inlet fluid temperature.

        """
        s = cls._longitudinal_position(xi_p, index, s_coefs)
        a_in = jnp.array(
            [cls._f1(s, beta_ij), -cls._f2(s, beta_ij)]
        )
        return a_in

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _general_solution_a_out(cls, xi_p: float, index: int, beta_ij: Array, s_coefs: Array) -> Array:
        """Outlet coefficient to evaluate the general solution.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        array
            (2,) array of coefficients for the outlet fluid
            temperature.

        """
        s = cls._longitudinal_position(xi_p, index, s_coefs)
        a_out = jnp.array(
            [cls._f2(s, beta_ij), cls._f3(s, beta_ij)]
        )
        return a_out

    @classmethod
    @partial(jit, static_argnames=['cls', 'order'])
    def _general_solution_a_b(cls, xi_p: float, index: int, beta_ij: Array, s_coefs: Array, J_coefs: Array, psi_coefs: Array, order: int = 101) -> Array:
        """Borehole wall coefficient to evaluate the general solution.

        Parameters
        ----------
        xi_p : float
            Coordinate along the borehole segment.
        index : int
            Index of the borehole segment.
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
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
            (2, `n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        # Initialize coefficients
        n_pipes = jnp.shape(beta_ij)[0]
        n_segments = jnp.shape(s_coefs)[1]
        n_nodes = jnp.shape(psi_coefs)[0]
        a_b = jnp.zeros((n_pipes, n_segments, n_nodes))
        # Longitudinal position corresponding to coordinate `xi_p`
        s = cls._longitudinal_position(xi_p, index, s_coefs)

        def general_solution_integrand(_xi_p: float, _i: int) -> Array:
            """Integrand of the general solution."""
            # Longitudinal positions, norms of Jacobian and basis functions
            # at integration points
            _t = cls._longitudinal_position(_xi_p, _i, s_coefs)
            _J = cls._norm_of_jacobian(_xi_p, _i, J_coefs)
            _psi = Basis._f_psi(_xi_p, psi_coefs)
            # Integrand
            _integrand = _J * jnp.stack(
                [
                    _psi * cls._f4(s - _t, beta_ij),
                    -_psi * cls._f5(s - _t, beta_ij)
                ]
            )
            return _integrand

        def _integral(_i: int, _b: float) -> Array:
            """Integrals over a portion of a segment."""
            _a_b = quadgl(
                partial(general_solution_integrand, _i=_i),
                points=jnp.array([-1., _b]),
                order=order
            )
            return _a_b

        def _zeros(_i: int) -> Array:
            """Array of zeros for segments below `index`."""
            return jnp.zeros((n_pipes, n_nodes))

        branches = [
            partial(_integral, _b=1.),
            partial(_integral, _b=xi_p),
            _zeros
        ]

        # Evaluation of the integral along all segments
        def _evaluate_integrals(_i, _a_b):
            _a_b = _a_b.at[:, _i, :].set(switch(_i - index + 1, branches, _i))
            return _a_b
        a_b = fori_loop(0, n_segments, _evaluate_integrals, a_b, unroll=False)

        return a_b.reshape(n_pipes, -1)

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _outlet_fluid_temperature_a_in(cls, beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        mixing : array
            (`n_pipes`/2,) array of coefficients to evaluate the outlet fluid
            temperature from the fluid temperatures at the top-end of the
            borehole in the upward flowing pipes following the relation:
            ``T_f_out = mixing @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
            the borehole segments as a function of `xi_p`.

        Returns
        -------
        float
            Coefficient for the inlet fluid temperature.

        """
        n_segments = jnp.shape(s_coefs)[1]
        L = cls._longitudinal_position(1., n_segments, s_coefs)
        a_in = (
            cls._f1(L, beta_ij) + cls._f2(L, beta_ij)
        ) / (cls._f3(L, beta_ij) - cls._f2(L, beta_ij))
        return a_in

    @classmethod
    @partial(jit, static_argnames=['cls', 'order'])
    def _outlet_fluid_temperature_a_b(cls, beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array, J_coefs: Array, psi_coefs: Array, order: int = 101) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        mixing : array
            (`n_pipes`/2,) array of coefficients to evaluate the outlet fluid
            temperature from the fluid temperatures at the top-end of the
            borehole in the upward flowing pipes following the relation:
            ``T_f_out = mixing @ T_fu``.
        s_coefs : array
            (`n_nodes`, `n_segments`,) array of polynomial coefficients
            for the evaluation of the longitudinal position (in meters) along
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
            (`n_nodes`,) array of coefficients for the borehole wall
            temperature.

        """
        n_segments = jnp.shape(s_coefs)[1]
        L = cls._longitudinal_position(1., n_segments, s_coefs)
        one_over_f3_minus_f2 = 1 / (cls._f3(L, beta_ij) - cls._f2(L, beta_ij))

        def outlet_fluid_temperature_integrand(_xi_p: float, _i: int) -> Array:
            """Integrand function."""
            # Longitudinal positions, norms of Jacobian and basis functions
            # at integration points
            _t = cls._longitudinal_position(_xi_p, _i, s_coefs)
            _J = cls._norm_of_jacobian(_xi_p, _i, J_coefs)
            _psi = Basis._f_psi(_xi_p, psi_coefs)
            # Integrand
            _integrand = _psi * _J * one_over_f3_minus_f2 * (
                cls._f4(L - _t, beta_ij)
                + cls._f5(L - _t, beta_ij)
            )
            return _integrand

        def outlet_fluid_temperature_integral(_i: int) -> Array:
            """Integrals over a segment."""
            _integral = quadgl(
                partial(outlet_fluid_temperature_integrand, _i= _i),
                order=order
            )
            return _integral

        a_b = vmap(
            outlet_fluid_temperature_integral,
            in_axes=0,
            out_axes=0
        )(jnp.arange(n_segments))

        return a_b.flatten()

    @staticmethod
    def _beta(beta_ij: Array) -> float:
        """Coefficient ``beta`` from Hellström (1991).

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        float

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta = 0.5 * (beta2 - beta1)
        return beta

    @staticmethod
    def _gamma(beta_ij: Array) -> float:
        """Coefficient ``gamma`` from Hellström (1991).

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        float

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta12 = beta_ij[0, 1]
        gamma = jnp.sqrt(0.25 * (beta1 + beta2)**2 + beta12 * (beta1 + beta2))
        return gamma

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _delta(cls, beta_ij: Array) -> float:
        """Coefficient ``delta`` from Hellström (1991).

        Parameters
        ----------
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        float

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta12 = beta_ij[0, 1]
        gamma = cls._gamma(beta_ij)
        delta = 1. / gamma * (beta12 + 0.5 * (beta1 + beta2))
        return delta

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _f1(cls, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f1`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = cls._beta(beta_ij)
        gamma = cls._gamma(beta_ij)
        delta = cls._delta(beta_ij)
        f1 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) - delta * jnp.sinh(gamma * s))
        return f1

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _f2(cls, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f2`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = cls._beta(beta_ij)
        gamma = cls._gamma(beta_ij)
        beta12 = beta_ij[0, 1]
        f2 = jnp.exp(beta * s) * beta12 / gamma * jnp.sinh(gamma * s)
        return f2

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _f3(cls, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f3`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta = cls._beta(beta_ij)
        gamma = cls._gamma(beta_ij)
        delta = cls._delta(beta_ij)
        f3 = jnp.exp(beta * s) * (jnp.cosh(gamma * s) + delta * jnp.sinh(gamma * s))
        return f3

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _f4(cls, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f4`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta12 = beta_ij[0, 1]
        beta = cls._beta(beta_ij)
        gamma = cls._gamma(beta_ij)
        delta = cls._delta(beta_ij)
        f4 = jnp.exp(beta * s) * (beta1 * jnp.cosh(gamma * s) - (delta * beta1 + beta2 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f4

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _f5(cls, s: Array | float, beta_ij: Array) -> Array | float:
        """Function ``f5`` from Hellström (1991).

        Parameters
        ----------
        s : array or float
            (M,) array of longitudinal positions (in meters).
        beta_ij: array
            (2, 2,) array of thermal conductance coefficients.

        Returns
        -------
        array or float
            (M,) array.

        """
        beta1 = beta_ij[0, 0]
        beta2 = beta_ij[1, 1]
        beta12 = beta_ij[0, 1]
        beta = cls._beta(beta_ij)
        gamma = cls._gamma(beta_ij)
        delta = cls._delta(beta_ij)
        f5 = jnp.exp(beta * s) * (beta2 * jnp.cosh(gamma * s) + (delta * beta2 + beta1 * beta12 / gamma) * jnp.sinh(gamma * s))
        return f5
