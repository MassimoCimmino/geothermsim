# -*- coding: utf-8 -*-
from functools import partial
from time import perf_counter
from typing import Tuple

from jax import numpy as jnp
from jax import Array, debug, jit, vmap
from jax.lax import cond, fori_loop
from jax.scipy.linalg import block_diag
from jax.typing import ArrayLike

from ..borefield.network import Network
from .load_history_reconstruction import LoadHistoryReconstruction


class gFunction:
    """g-Function.

    Parameters
    ----------
    borefield: network
        The borefield.
    m_flow : float or array
        Total fluid mass flow rate (in kg/s), or (`n_boreholes`,) array of
        fluid mass flow rate per borehole.
    time : array_like
        Times (in seconds) to evaluate the g-function.
    alpha : float
        Ground thermal diffusivity (in m^2/s)
    k_s : float
        Ground thermal conductivity (in W/m-K).
    p : array_like or None, default: ``None``
        (`n_points`, 3,) array of positions to evaluate the ground
        temperature. If `p` is ``None``, the ground temperature is not
        evaluated.
    disp : bool, default: ``True``
        Set to ``True`` to print initialization progression messages.

    Attributes
    ----------
    m_flow_network : float
        Total fluid mass flow rate (in kg/s).
    g : array
        The g-function of the borefield.
    q : array
        The heat extraction rate at the nodes (in W/m).
    T_b : array
        The borehole wall temperature at the nodes (in degree Celsius).
    T_f_in : array
        The inlet fluid temperature (in degree Celsius).
    T : array
        The ground temperature (in degree Celsius) at positions `p`.

    """

    def __init__(self, borefield: Network, m_flow: float | Array, time: ArrayLike, alpha: float, k_s: float, p: ArrayLike | None = None, disp: bool = True):
        # Runtime type validation
        if not isinstance(time, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {time}")
        if not isinstance(p, ArrayLike) and p is not None:
            raise TypeError(f"Expected arraylike or None input; got {p}")
        # Convert input to jax.Array
        time = jnp.asarray(time)
        if p is not None:
            p = jnp.asarray(p)

        # --- Class atributes ---
        # Parameters
        self.borefield = borefield
        self.n_nodes = self.borefield.n_boreholes * self.borefield.n_nodes
        self.m_flow = m_flow
        self.time = time
        self.time = jnp.concatenate((jnp.array([0.]), self.time))
        self.alpha = alpha
        self.k_s = k_s
        self.p = p
        # Additional attributes
        self.n_times = len(time)
        if p is None:
            self.T = None
            self.n_points = 0
        else:
            self.n_points = p.shape[0]
        if len(jnp.shape(m_flow)) == 0:
            self.m_flow_network = m_flow
        else:
            self.m_flow_network = m_flow.sum()

        # --- Initialization ---
        self.loaHisRec = LoadHistoryReconstruction(
            borefield, time, alpha, p=p, disp=disp)
        self.initialize_systems_of_equations()

    def initialize_systems_of_equations(self):
        """Initialize the system of equations.

        """
        N = self.n_nodes
        # Thermal response factors
        self.h_to_self = jnp.concatenate(
            (
                jnp.zeros(
                    (1,
                     self.borefield.n_boreholes,
                     self.borefield.n_nodes,
                     self.borefield.n_boreholes,
                     self.borefield.n_nodes)
                ),
                self.loaHisRec.h_to_self / (2 * jnp.pi * self.k_s)
            ),
            axis=0)
        # Borehole heat transfer rate coefficients
        cp_f = self.borefield.fluid.specific_heat()
        self.g_in, self.g_b = self.borefield._heat_extraction_rate_to_self(self.m_flow, cp_f)
        # Initialize system of equations
        self.A = jnp.block(
            [[jnp.eye(N), self.g_in.reshape((-1, 1))],
             [self.borefield.w.flatten(), jnp.zeros((1, 1))]]
            )
        self.B = jnp.zeros(N + 1)
        # Apply constant total heat extraction rate
        self.B = self.B.at[-1].set(2 * jnp.pi * self.k_s * self.borefield.L.sum())

    def update_system_of_equations(self, h_to_self: Array, T0: Array) -> Tuple[Array, Array]:
        """Update the system of equations.

        Parameters
        ----------
        h_to_self : array
            Array of thermal response factors.
        T0 : array
            Borehole wall temperature at nodes assuming zero heat
            extraction rate.

        Returns
        -------
        A : array
            Linear system of equations.
        B : array
            Right-hand side of the linear system of equations.

        """
        N = self.n_nodes
        # Update system of equations for thermal response factors at the
        # current time step `h_to_self`
        A = self.A.at[:N, :N].set(jnp.eye(N) + jnp.einsum('iml,iljn->imjn', self.g_b, h_to_self).reshape((N, N)))
        # Apply current borehole wall temperature at nodes `T0`
        B = self.B.at[:N].set(vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(-self.g_b, T0.reshape((self.borefield.n_boreholes, -1))).flatten())
        return A, B

    def simulate(self, disp: bool = True, print_every: int = 10):
        """Evaluate the g-function.

        Parameters
        ----------
        disp : bool, default: ``True``
            Set to ``True`` to print simulation progression messages.
        print_every : int, default: ``10``
            Period at which simulation progression messages are printed.

        """
        tic = perf_counter()
        next_k = print_every
        if disp:
            print('Simulation start.')
        self.loaHisRec.reset_history()
        # Initialize arrays
        self.q = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_b = jnp.zeros((self.n_times, self.borefield.n_boreholes, self.borefield.n_nodes))
        self.T_f_in = jnp.zeros(self.n_times)
        if self.p is not None:
            self.T = jnp.zeros((self.n_times, self.n_points))

        def simulate_step(
            i: int,
            val: Tuple[Array, Array, Array, Array, Array | None]
        ) -> Tuple[Array, Array, Array, Array, Array | None]:
            """Single simulation step."""
            # Unpack values
            time, q, T_b, T_f_in, T = val
            # Time step
            dtime = time[i + 1] - time[i]
            # Temporal and spatial superposition of past loads
            _q_reconstructed = self.loaHisRec._reconstruct_load_history(
                time[i + 1], time[1:], q)
            T0 = self.loaHisRec._temperature(
                self.loaHisRec.h_to_self,
                _q_reconstructed
            ) / (2 * jnp.pi * self.k_s)
            # Build and solve system of equations
            _q, _T_b, _T_f_in = self._simulate_step(dtime, T0)
            # Apply latest heat extraction rates
            q = self.loaHisRec._current_load(q, _q, i)
            # Store results
            T_b = T_b.at[i].set(_T_b)
            T_f_in = T_f_in.at[i].set(_T_f_in)
            # Evaluate ground temperatures
            if self.p is not None:
                _q_reconstructed = self.loaHisRec._reconstruct_load_history(
                    time[i + 1], time[1:], q)
                T = T.at[i].set(
                    self.loaHisRec._temperature_to_point(
                        self.loaHisRec.h_to_point,
                        _q_reconstructed
                    ) / (2 * jnp.pi * self.k_s)
                )
            if disp:
                toc = perf_counter()
                cond(
                    (i + 1) % print_every == 0,
                    lambda _: debug.print(
                        'Completed {i} of {n_times} time steps. ',
                        i=i+1, n_times=self.n_times
                    ),
                    lambda _: None,
                    None
                )
            return time, q, T_b, T_f_in, T
        # Pack variables
        val = self.time, self.q, self.T_b, self.T_f_in, self.T
        # Run simulation
        _, self.q, self.T_b, self.T_f_in, self.T = fori_loop(
            0, self.n_times, simulate_step, val, unroll=False)

        # Outlet fluid temperature
        cp_f = self.borefield.fluid.specific_heat()
        T_f_out = self.T_f_in - 2 * jnp.pi * self.k_s * self.borefield.L.sum() / (self.m_flow_network * cp_f)
        # Average fluid temperature
        T_f = 0.5 * (self.T_f_in + T_f_out)
        # Borefield thermal resistance
        R_field = self.borefield._effective_borefield_thermal_resistance(self.m_flow, cp_f)
        # Effective borehole wall temperature
        self.g = T_f - 2 * jnp.pi * self.k_s * R_field
        if disp:
            toc = perf_counter()
            debug.print(
                'Simulation end. Elapsed time: {clock:.2f} seconds.',
                clock=toc-tic
            )

    @partial(jit, static_argnames=['self'])
    def _simulate_step(self, dtime: float, T0: Array) -> Tuple[Array, Array, float]:
        """Solve a single time step.

        Parameters
        ----------
        dtime : float
            Current time step variation (in seconds).
        T0 : array
            Borehole wall temperature at nodes assuming zero heat
            extraction rate.

        Returns
        -------
        q : array
            The heat extraction rate at the nodes (in W/m).
        T_b : array
            The borehole wall temperature at the nodes (in degree
            Celsius).
        T_f_in : float
            The inlet fluid temperature (in degree Celsius).

        """
        # Current thermal response factors
        h_shape = self.h_to_self.shape
        h = vmap(
            lambda _h: jnp.interp(dtime, self.time, _h),
            in_axes=-1,
            out_axes=-1
            )(
            self.h_to_self.reshape((h_shape[0], -1))
        ).reshape(h_shape[1:])
        # Build and solve system of equations
        A, B = self.update_system_of_equations(h, T0)
        X = jnp.linalg.solve(A, B)
        q = X[:self.n_nodes].reshape((self.borefield.n_boreholes, -1))
        T_b = T0 + jnp.tensordot(h, q, axes=([-2, -1], [-2, -1]))
        T_f_in = X[-1]
        return q, T_b, T_f_in
        
