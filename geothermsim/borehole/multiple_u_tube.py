# -*- coding: utf-8 -*-
from functools import partial
from typing import Tuple

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import fori_loop, switch
from jax.scipy.linalg import expm

from ._tube import _Tube
from ..basis import Basis
from ..utilities import quadgl


class MultipleUTube(_Tube):
    """Multiple U-Tube geothermal borehole.

    Parameters
    ----------
    R_d : array_like or callable
        (`n_pipes`, `n_pipes`,) array of thermal resistances (in m-K/W),
        or callable that takes the mass flow rate as input (in kg/s) and
        returns a (`n_pipes`, `n_pipes`,) array.
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
        top_connectivity : tuple of array
            Tuple of two arrays (``c_in`` and ``c_u``) of shape
            (`n_pipes`/2,) and (`n_pipes`/2, `n_pipes`/2,). The two
            arrays give the relation between the inlet fluid temperature,
            the fluid temperatures at the top-end of the borehole in the
            upward flowing pipes, and the fluid temperatures at the top-end
            of the borehole in the downward flowing pipes following the
            relation: ``T_fd = c_in * T_f_in + c_u @ T_fu``.
        beta_ij : array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.
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
        n_pipes = jnp.shape(beta_ij)[0]
        n_pipes_over_two = n_pipes // 2
        n_segments = jnp.shape(s_coefs)[1]
        # General solution at coordinates `xi`
        E_0_xi = cls._general_solution_a_0(
            xi_p, index, beta_ij, s_coefs)
        # General solution at the bottom-end
        E_0 = cls._general_solution_a_0(
            1., n_segments, beta_ij, s_coefs)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in, c_u = top_connectivity

        # Intermediate coefficients
        delta_E_0_xi = (
            E_0_xi[:, :n_pipes_over_two] @ c_u
            + E_0_xi[:, n_pipes_over_two:]
        )
        delta_E_0_u = (
            E_0[:n_pipes_over_two]
            - E_0[n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :n_pipes_over_two] @ c_u
            + delta_E_0_u[:, n_pipes_over_two:]
        )
        delta_E_0_in = (
            E_0[n_pipes_over_two:]
            - E_0[:n_pipes_over_two]
        )
        delta_E_0_in = delta_E_0_in[:, :n_pipes_over_two] @ c_in

        # Inlet coefficient to evaluate the fluid temperatures
        a_in = (
            delta_E_0_xi @ jnp.linalg.solve(delta_E_0_u, delta_E_0_in)
            + E_0_xi[:, :n_pipes_over_two] @ c_in
        )
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

        """
        n_pipes = jnp.shape(beta_ij)[0]
        n_pipes_over_two = n_pipes // 2
        n_segments = jnp.shape(s_coefs)[1]
        # General solution at coordinates `xi`
        E_0_xi = cls._general_solution_a_0(
            xi_p, index, beta_ij, s_coefs)
        E_b_xi = cls._general_solution_a_b(
            xi_p, index, beta_ij, s_coefs, J_coefs, psi_coefs, order)
        # General solution at the bottom-end
        E_0 = cls._general_solution_a_0(
            1., n_segments, beta_ij, s_coefs)
        E_b = cls._general_solution_a_b(
            1., n_segments, beta_ij, s_coefs, J_coefs, psi_coefs, order)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in, c_u = top_connectivity

        # Intermediate coefficients
        delta_E_0_xi = (
            E_0_xi[:, :n_pipes_over_two] @ c_u
            + E_0_xi[:, n_pipes_over_two:]
        )
        delta_E_0_u = (
            E_0[:n_pipes_over_two]
            - E_0[n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :n_pipes_over_two] @ c_u
            + delta_E_0_u[:, n_pipes_over_two:]
        )
        delta_E_b = (
            E_b[n_pipes_over_two:]
            - E_b[:n_pipes_over_two]
        )

        # Borehole wall coefficient to evaluate the fluid temperatures
        a_b = (
            delta_E_0_xi @ jnp.linalg.solve(delta_E_0_u, delta_E_b)
            + E_b_xi
        )
        return a_b

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _general_solution_a_0(cls, xi_p: float, index: int, beta_ij: Array, s_coefs: Array) -> Array:
        """Top-end coefficient to evaluate the general solution.

        Parameters
        ----------
        xi : float
            Coordinates along the borehole.
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`) array of coefficients for the
            top-end fluid temperature.

        """
        s = cls._longitudinal_position(xi_p, index, s_coefs)
        A = cls._ode_coefficients(beta_ij)
        a_0 = cls._phi(s, A)
        return a_0

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
            (`n_pipes`, `n_nodes`,) array of coefficients for the
            borehole wall temperature.

        """
        # Initialize coefficients
        n_pipes = jnp.shape(beta_ij)[0]
        n_segments = jnp.shape(s_coefs)[1]
        n_nodes = jnp.shape(psi_coefs)[0]
        a_b = jnp.zeros((n_pipes, n_segments, n_nodes))
        # Longitudinal position corresponding to coordinate `xi_p`
        s = cls._longitudinal_position(xi_p, index, s_coefs)
        # Coefficients of the ODE
        A = cls._ode_coefficients(beta_ij)
        b = -A.sum(axis=1)

        def general_solution_integrand(_xi_p: float, _i: int) -> Array:
            """Integrand of the general solution."""
            # Longitudinal positions, norms of Jacobian and basis functions
            # at integration points
            _t = cls._longitudinal_position(_xi_p, _i, s_coefs)
            _J = cls._norm_of_jacobian(_xi_p, _i, J_coefs)
            _psi = Basis._f_psi(_xi_p, psi_coefs)
            # Integrand
            _phi = cls._phi(s - _t, A)
            _phi_b = _phi @ b
            _phi_b_psi = jnp.outer(_phi_b, _psi)
            _integrand = _J * _phi_b_psi
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

    @staticmethod
    @jit
    def _ode_coefficients(beta_ij: Array) -> Array:
        """Coefficients of the ordinary differential equations.

        Parameters
        ----------
        beta_ij: array
            (`n_pipes`, `n_pipes`,) array of thermal conductance
            coefficients.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients of the ordinary
            differential equations.

        """
        n_pipes_over_two = jnp.shape(beta_ij)[0] // 2
        diag_indices = jnp.diag_indices_from(beta_ij)
        ode_coefficients = beta_ij.at[diag_indices].set(
            -beta_ij.sum(axis=1)
        )
        ode_coefficients = ode_coefficients.at[n_pipes_over_two:, :].multiply(-1)
        return ode_coefficients

    @classmethod
    @partial(jit, static_argnames=['cls'])
    def _outlet_fluid_temperature_a_in(cls, beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array) -> float:
        """Inlet coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
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
        n_pipes = jnp.shape(beta_ij)[0]
        n_pipes_over_two = n_pipes // 2
        n_segments = jnp.shape(s_coefs)[1]
        # General solution at the bottom-end
        E_0 = cls._general_solution_a_0(
            1., n_segments, beta_ij, s_coefs)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in, c_u = top_connectivity
        # Coefficient for mixing of the fluid at the outlet
        # T_f_out = m_u @ T_f_u(-1)
        m_u = mixing

        # Intermediate coefficients
        delta_E_0_u = (
            E_0[:n_pipes_over_two]
            - E_0[n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :n_pipes_over_two] @ c_u
            + delta_E_0_u[:, n_pipes_over_two:]
        )
        delta_E_0_in = (
            E_0[n_pipes_over_two:]
            - E_0[:n_pipes_over_two]
        )
        delta_E_0_in = delta_E_0_in[:, :n_pipes_over_two] @ c_in

        # Inlet coefficient to evaluate the outlet fluid temperature
        a_in = m_u @ jnp.linalg.solve(delta_E_0_u, delta_E_0_in)
        return a_in

    @classmethod
    @partial(jit, static_argnames=['cls', 'order'])
    def _outlet_fluid_temperature_a_b(cls, beta_ij: Array, top_connectivity: Tuple[Array, Array], mixing: Array, s_coefs: Array, J_coefs: Array, psi_coefs: Array, order: int = 101) -> Array:
        """Borehole coefficient to evaluate the outlet fluid temperature.

        Parameters
        ----------
        beta_ij: array
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
        n_pipes = jnp.shape(beta_ij)[0]
        n_pipes_over_two = n_pipes // 2
        n_segments = jnp.shape(s_coefs)[1]
        # General solution at the bottom-end
        E_0 = cls._general_solution_a_0(
            1., n_segments, beta_ij, s_coefs)
        E_b = cls._general_solution_a_b(
            1., n_segments, beta_ij, s_coefs, J_coefs, psi_coefs, order)
        # Coefficients for connectivity at the top of the borehole
        # T_f_d(-1) = c_in * T_f_in + c_u @ T_f_u(-1)
        c_in, c_u = top_connectivity
        # Coefficient for mixing of the fluid at the outlet
        # T_f_out = m_u @ T_f_u(-1)
        m_u = mixing

        # Intermediate coefficients
        delta_E_0_u = (
            E_0[:n_pipes_over_two]
            - E_0[n_pipes_over_two:]
        )
        delta_E_0_u = (
            delta_E_0_u[:, :n_pipes_over_two] @ c_u
            + delta_E_0_u[:, n_pipes_over_two:]
        )
        delta_E_b = (
            E_b[n_pipes_over_two:]
            - E_b[:n_pipes_over_two]
        )

        # Borehole coefficient to evaluate the outlet fluid temperature
        a_b = m_u @ jnp.linalg.solve(delta_E_0_u, delta_E_b)
        return a_b

    @staticmethod
    @jit
    def _phi(s: float, ode_coefficients: Array) -> Array:
        """State-transition matrix.

        Parameters
        ----------
        s : float
            Longitudinal position along the borehole (in meters).
        ode_coefficients : array
            (`n_pipes`, `n_pipes`,) array of coefficients of the ordinary
            differential equation.

        Returns
        -------
        array
            State-trasition matrix of shape (`n_pipes`, `n_pipes`,).

        """
        return expm(ode_coefficients * s)
