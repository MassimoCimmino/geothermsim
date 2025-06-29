# -*- coding: utf-8 -*-
from functools import partial

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.lax import fori_loop
from jax.typing import ArrayLike

from ..utilities import comb


class Multipole:
    """Multipole method for 3d heat transfer.

    Parameters
    ----------
    r_b : float
        Borehole radius (in meters).
    r_p : array_like
        Outer radius of the pipes (in meters).
    p : array_like
        (`n_pipes`, 2) array of pipe positions (in meters). The position
        ``(0, 0)`` corresponds to the axis of the borehole.
    k_s : float
        Ground thermal conductivity (in W/m-K).
    k_b : float
        Grout thermal conductivity (in W/m-K).
    J : int, default: 3
        Order of the multipole solution.

    Attributes
    ----------
    n_pipes : int
        Number of pipes.
    sigma : float
        Dimensionless parameter for the two thermal conductivities.
    z : complex array
        (`n_pipes`,) complex array of pipe positions (in meters).

    """

    def __init__(self, r_b: float, r_p: ArrayLike, p: ArrayLike, k_s: float, k_b: float, J: int = 3):
        # Runtime type validation
        if not isinstance(r_p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {r_p}")
        if not isinstance(p, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {p}")
        # Convert input to jax.Array
        r_p = jnp.asarray(r_p)
        p = jnp.atleast_2d(p)

        # --- Class atributes ---
        # Parameters
        self.r_b = r_b
        self.r_p = r_p
        self.p = p
        self.k_s = k_s
        self.k_b = k_b
        self.J = J
        # Additional attributes
        self.n_pipes = len(r_p)
        self.sigma = (k_b - k_s) / (k_b + k_s)
        self.z = p[:, 0] + 1.j * p[:, 1]
        self._zz = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )(self.z.conj(), self.z)
        self._dis = vmap(
            jnp.add,
            in_axes=(0, None),
            out_axes=0
            )(self.z, -self.z)

        # Initialize mutipole coefficients and resistances
        self._initialize_factors()
        self._initialize_thermal_resistances()
        self._initialize_system_of_equations()

    def beta(self, R_fp: Array) -> Array:
        """Dimensionless fluid to pipe thermal resistances.

        Parameters
        ----------
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            Dimensionless fluid to pipe thermal resistances.

        """
        return R_fp * (2 * jnp.pi * self.k_b)

    @partial(jit, static_argnames=['self'])
    def fluid_temperatures(self, q: Array, T_b: float, R_fp: Array) -> Array:
        """Evaluate fluid temperatures.

        Parameters
        ----------
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        T_b : float
            Average borehole wall temperature (in degree Celsius).
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            (`n_pipes`,) array of Fluid temperatures (in degree Celsius).

        """
        # Dimensionless fluid to pipe thermal resistances
        beta = self.beta(R_fp)
        # Multipole strengths (eqs. 34-36)
        P = self.multipoles(q, beta)
        # Fluid temperatures (eq. 32)
        a_q = self._thermal_resistances_line(beta)
        a_P = self.thermal_resistances_multipole
        T_f = T_b + a_q @ q + jnp.tensordot(a_P, P).real
        return T_f

    @partial(jit, static_argnames=['self'])
    def multipoles(self, q: Array, beta: Array) -> Array:
        """Evaluate multipoles.

        Parameters
        ----------
        q : array
            (`n_pipes`,) array of heat injection rates in each pipe
            (in W/m).
        beta : array
            Dimensionless fluid to pipe thermal resistances.

        Returns
        -------
        array
            (`n_pipes`, `J`,) array of multipole strengths.

        """
        N = self.n_pipes * self.J
        k = jnp.arange(1, self.J + 1)
        # Coefficients that depend on the fluid to pipe thermal
        # resistances (eq. 36)
        beta_k = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )(beta, k)
        coeffs = (1 + beta_k) / (1 - beta_k)
        diag_indices = jnp.diag_indices_from(self.A)

        def update_matrix(A, diag_indices, coeffs):
            update_values = jnp.concatenate([coeffs.flatten(), -coeffs.flatten()])
            # diag_indices should be a tuple of arrays (e.g., (row_indices, col_indices))
            return A.at[diag_indices].add(update_values)

        A = update_matrix(self.A, diag_indices, coeffs)

        # Right hand side (eq. 36)
        B = (-self.factors_line @ q).flatten()
        B = jnp.concatenate([B.real, B.imag])
        # Solve for multipoles (eq. 36)
        P = jnp.linalg.solve(A, B)
        P = (P[:N] + 1.j * P[N:]).reshape((self.n_pipes, self.J))
        return P

    @partial(jit, static_argnames=['self'])
    def thermal_resistances(self, R_fp: Array) -> Array:
        """Evaluate delta-circuit thermal resistances.

        Parameters
        ----------
        R_fp : array
            (`n_pipes`,) array of fluid to pipe thermal resistance
            (in m-K/W).

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of delta-circuit thermal
            resistances (in m-K/W).

        """
        # Evaluate fluid temperatures for unit heat transfer rates at
        # each pipe, individually (eq. 32).
        R = vmap(
            self.fluid_temperatures,
            in_axes=(0, None, None),
            out_axes=0
            )(jnp.eye(self.n_pipes), 0., R_fp)
        # Thermal conductances (eq. 51)
        K = -jnp.linalg.inv(R)
        # Delta-circuit thermal resistances (eq. 51)
        diag_indices = jnp.diag_indices_from(K)

        def set_diagonal_to_negative_row_sums(K, diag_indices):
            row_sums = K.sum(axis=1)
            return K.at[diag_indices].set(-row_sums)

        n = K.shape[0]
        diag_indices = (jnp.arange(n), jnp.arange(n))
        K = set_diagonal_to_negative_row_sums(K, diag_indices)
        R_d = 1 / K
        return R_d

    def _factors_line(self, k: int) -> Array:
        """Multipole factor coefficients for the heat injection rates.

        Parameters
        ----------
        k : int
            Target multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients

        """
        z = self.z
        zz = self._zz
        I = jnp.eye(self.n_pipes)
        # First term (eq. 34)
        denom_1 = ((1 - I) * -self._dis + I)**k
        numer_1 = ((1 - I) * self.r_p**k).T
        # Second term (eq. 34)
        denom_2 = (self.r_b**2 - zz.T)**k
        numer_2 = self.sigma * vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )(self.r_p**k, z.conj()**k)
        # Sum of the two terms
        a_q = (numer_1 / denom_1 + numer_2 / denom_2) / k / (2 * jnp.pi * self.k_b)
        return a_q

    def _factors_multipole(self, k: int, j: int) -> Array:
        """Multipole factor coefficients for the multipoles.

        Parameters
        ----------
        k : int
            Target multipole order.
        J : int
            Source multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients

        """
        I = jnp.eye(self.n_pipes)
        # Third term (eq. 34)
        numerator = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )((-self.r_p)**k, self.r_p**j)
        denominator = self._dis**(k + j) + I
        a_P = (1 - I) * comb(k + j - 1, j - 1) * numerator / denominator
        return a_P

    def _factors_conjugate_multipole(self, k: int, j: int) -> Array:
        """Multipole factor coefficients for the conjugate multipoles.

        Parameters
        ----------
        k : int
            Target multipole order.
        J : int
            Source multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of coefficients

        """
        j_p_max = jnp.minimum(k, j)
        z = self.z
        # Fourth term (eq. 34)
        # This is a for loop but uses jax.lax.fori_loop to allow tracing.
        a_P_conj = jnp.zeros((self.n_pipes, self.n_pipes), dtype=complex)
        def body_fun(i, val):
            numerator = vmap(
                jnp.multiply,
                in_axes=(0, None),
                out_axes=0
                )(self.r_p**k * z**(j - i), self.r_p**j * z.conj()**(k - i))
            denominator = (self.r_b**2 - self._zz.T)**(k + j - i)
            return val + self.sigma * comb(j, i) * comb(k + j - i - 1, j - 1) * numerator / denominator
        a_P_conj = fori_loop(0, j_p_max + 1, body_fun, a_P_conj)
        return a_P_conj

    def _initialize_factors(self):
        """Initialize multipole factor coefficients.

        """
        k = jnp.arange(1, self.J + 1)
        j = jnp.arange(1, self.J + 1)
        self.factors_line = vmap(
            self._factors_line,
            in_axes=0,
            out_axes=1
        )(k)
        self.factors_multipole = vmap(
            vmap(
                self._factors_multipole,
                in_axes=(0, None),
                out_axes=1
                ),
            in_axes=(None, 0),
            out_axes=-1
        )(k, j)
        self.factors_conjugate_multipole = vmap(
            vmap(
                self._factors_conjugate_multipole,
                in_axes=(0, None),
                out_axes=1
                ),
            in_axes=(None, 0),
            out_axes=-1
        )(k, j)

    def _initialize_system_of_equations(self):
        """Initialize the system of equations for multipoles.

        """
        N = self.n_pipes * self.J
        A = self.factors_multipole.reshape((N, N))
        C = self.factors_conjugate_multipole.reshape((N, N))
        # Split the real and image parts to allow the solution of a linear
        # system of equations (eq. 36). The coefficients `C` multiply the
        # conjugate of the multipoles.
        self.A = jnp.block(
            [[A.real + C.real, -A.imag + C.imag],
             [A.imag + C.imag, A.real - C.real]]
            )

    def _initialize_thermal_resistances(self):
        """Initialize thermal resistances.

        """
        self.thermal_resistances_line_zero_beta = (
            self._thermal_resistances_line_zero_beta()
            )
        self.thermal_resistances_multipole = vmap(
            self._thermal_resistances_multipole,
            in_axes=0,
            out_axes=-1
            )(jnp.arange(1, self.J + 1))

    def _thermal_resistances_line(self, beta: Array) -> Array:
        """Evaluate thermal resistances for heat extraction rates.

        Parameters
        ----------
        beta : array
            Dimensionless fluid to pipe thermal resistances.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances
            (in m-K/W).

        """
        diag_indices = jnp.diag_indices_from(
            self.thermal_resistances_line_zero_beta
            )

        def update_thermal_resistances(R, diag_indices, beta, k_b):
            update_values = beta / (2 * jnp.pi * k_b)
            return R.at[diag_indices].add(update_values)
        R = update_thermal_resistances(
            self.thermal_resistances_line_zero_beta,
            diag_indices,
            beta,
            self.k_b
        )
        return R

    def _thermal_resistances_line_zero_beta(self) -> Array:
        """Initialize thermal resistances for heat extraction rates.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances
            (in m-K/W).

        """
        z = self.z
        zz = self._zz
        I = jnp.eye(self.n_pipes)
        absdis = jnp.abs(self._dis)
        # Thermal resistances (eq. 33), assuming `beta=0`
        R = 1 / (2 * jnp.pi * self.k_b) * (
            jnp.log(self.r_b / ((1 - I) * absdis + jnp.diag(self.r_p)))
            + self.sigma * jnp.log(
                self.r_b**2 / (
                    (1 - I) * jnp.abs(self.r_b**2 - zz)
                    + jnp.diag(self.r_b**2 - jnp.abs(z)**2)
                )
            )
        )
        return R

    def _thermal_resistances_multipole(self, j: int) -> Array:
        """Evaluate thermal resistances for multipoles.

        Parameters
        ----------
        j : int
            Multipole order.

        Returns
        -------
        array
            (`n_pipes`, `n_pipes`,) array of thermal resistances.

        """
        z = self.z
        zz = self._zz
        I = jnp.eye(self.n_pipes)
        # Thermal resistances of multipoles (eq. 32)
        numer_1 = (1 - I) * self.r_p**j
        denom_1 = self._dis**j + I
        numer_2 = vmap(
            jnp.multiply,
            in_axes=(0, None),
            out_axes=0
            )(z.conj()**j, self.r_p**j)
        denom_2 = (self.r_b**2 - zz)**j
        a_P = (numer_1 / denom_1 + self.sigma * numer_2 / denom_2)
        return a_P
