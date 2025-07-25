# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing import Tuple
from typing_extensions import Self

from jax import numpy as jnp
from jax import Array, jit, vmap
import numpy as np
from scipy.special import roots_legendre


def quad(fun: Callable[[float], Array | float], points: Array = jnp.array([-1., 1.]), order: int = 51, rule: str = 'gl') -> Array | float:
    """Numerical integral.

    Parameters
    ----------
    fun : callable
        Function to integrate. Takes a float as the only argument and
        returns either an array or a float.
    points : array, default: [-1., 1.]
        Points delimiting the subintervals for the integration.
    order : int, default: 51
        Order of the numerical quadrature.
    rule : {'gl', 'ts'}, default: 'gl'
        Quadrature rule. 'gl' corresponds to Gauss-Legendre quadrature
        and 'ts' to tanh-sinh quadrature.

    Returns
    -------
    array or float
        Intregal of `fun`.

    """
    if rule.lower() == 'gl' or rule.lower() == 'gauss-legendre':
        return quadgl(fun, points=points, order=order)
    if rule.lower() == 'ts' or rule.lower() == 'tanh-sinh':
        return quadts(fun, points=points, order=order)


def quadgl(fun: Callable[[float], Array | float], points: Array = jnp.array([-1., 1.]), order: int = 51) -> Array | float:
    """Numerical integral using Gauss-Legendre rule.

    Parameters
    ----------
    fun : callable
        Function to integrate. Takes a float as the only argument and
        returns either an array or a float.
    points : array, default: [-1., 1.]
        Points delimiting the subintervals for the integration.
    order : int, default: 51
        Order of the numerical quadrature.

    Returns
    -------
    array or float
        Intregal of `fun`.

    """
    x, w = _quadgl_weights(order)
    return _quad(fun, points, x, w)


def quadts(fun: Callable[[float], Array | float], points: Array = jnp.array([-1., 1.]), order: int = 51) -> Array | float:
    """Numerical integral using tanh-sinh rule.

    Parameters
    ----------
    fun : callable
        Function to integrate. Takes a float as the only argument and
        returns either an array or a float.
    points : array, default: [-1., 1.]
        Points delimiting the subintervals for the integration.
    order : int, default: 51
        Order of the numerical quadrature.

    Returns
    -------
    array or float
        Intregal of `fun`.

    """
    x, w = _quadts_weights(order)
    return _quad(fun, points, x, w)


def _fixed_quad(fun: Callable[[float], Array | float], a: float, b: float, x: Array, w: Array) -> Array | float:
    """Numerical integral on a single interval.

    Parameters
    ----------
    fun : callable
        Function to integrate. Takes a float as the only argument and
        returns either an array or a float.
    a, b : float
        Lower and upper limit of integration.
    x : array
        Quadrature points.
    w : array
        Quadrature weights.

    Returns
    -------
    array or float
        Intregal of `fun`.

    """
    x = 0.5 * (b + a) + 0.5 * (b - a) * x
    y = vmap(fun, in_axes=0, out_axes=-1)(x)
    return 0.5 * (b - a) * y @ w


def _quad(fun: Callable[[float], Array | float], points: Array, x: Array, w: Array) -> Array | float:
    """Numerical integral over subintervals.

    Parameters
    ----------
    fun : callable
        Function to integrate. Takes a float as the only argument and
        returns either an array or a float.
    points : array
        Points delimiting the subintervals for the integration.
    x : array
        Quadrature points.
    w : array
        Quadrature weights.

    Returns
    -------
    array or float
        Intregal of `fun`.

    """
    a, b = points[:-1], points[1:]
    integral = vmap(
        lambda _a, _b: _fixed_quad(fun, _a, _b, x, w),
        in_axes=(0, 0),
        out_axes=0
    )(a, b).sum(axis=0)
    return integral


def _quadgl_weights(order: int) -> Tuple[Array, Array]:
    """Points and weights of the Gauss-Legendre quadrature.

    Parameters
    ----------
    order : int
        Order of the tanh-sinh quadrature.

    Returns
    -------
    x : array
        Quadrature points.
    w : array
        Quadrature weights.

    """
    x, w = roots_legendre(order)
    x = jnp.asarray(x)
    w = jnp.asarray(w)
    return x, w


def _quadts_weights(order: int) -> Tuple[Array, Array]:
    """Points and weights of the tanh-sinh quadrature.

    Parameters
    ----------
    order : int
        Order of the tanh-sinh quadrature.

    Returns
    -------
    x : array
        Quadrature points.
    w : array
        Quadrature weights.

    """
    one_minus_eps = jnp.array(1.0) - 10 * jnp.finfo(jnp.array(1.0).dtype).eps
    t_max = jnp.arcsinh(2. / jnp.pi * jnp.arctanh(one_minus_eps))
    t = jnp.linspace(-t_max, t_max, num=order)

    x = jnp.tanh(0.5 * jnp.pi * jnp.sinh(t))
    w = 0.5 * jnp.pi * jnp.cosh(t) / jnp.cosh(0.5 * jnp.pi * jnp.sinh(t))**2
    w = w * jnp.diff(t)[0]
    w = 2 * w / jnp.sum(w)
    return x, w
