import math
from functools import partial

import numpy as np
import scipy

import jax
import jax.numpy as jnp
from jax import random
from jax import jit

from .norm import normalize_l2_cw
from .util import promote_arg_dtypes
from .array import hermitian

def gaussian_mtx(key, N, D, normalize_atoms=True):
    """A dictionary/sensing matrix where entries are drawn independently from normal distribution.

    Args:
        key: a PRNG key used as the random key.
        N (int): Number of rows of the sensing matrix 
        D (int): Number of columns of the sensing matrix
        normalize_atoms (bool): Whether the columns of sensing matrix are normalized 
          (default True)

    Returns:
        (jax.numpy.ndarray): A Gaussian sensing matrix of shape (N, D)

    Example:

        >>> from jax import random
        >>> import cr.sparse as crs
        >>> import cr.sparse.dict
        >>> m, n = 8, 16
        >>> Phi = cr.sparse.dict.gaussian_mtx(random.PRNGKey(0), m, n)
        >>> print(Phi.shape)
        (8, 16)
        >>> print(crs.norms_l2_cw(Phi))
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        >>> print(cr.sparse.dict.coherence(Phi))
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        >>> print(cr.sparse.dict.babel(Phi))
        [0.85866616 1.59791754 2.13943785 2.61184779 2.9912899  3.38281051
        3.74641682 4.08225813 4.29701559 4.49942648 4.68680188 4.83106192
        4.95656728 5.05541184 5.10697535]
    """
    shape = (N, D)
    dict = random.normal(key, shape)
    if normalize_atoms:
        dict = normalize_l2_cw(dict)
    else:
        sigma = math.sqrt(N)
        dict = dict / sigma
    return dict


def _pascal_lower(n):
    A = jnp.empty((n, n), dtype=jnp.int32)
    A = A.at[0, :].set(0)
    A = A.at[:, 0].set(1)
    for i in range(1, n):
        for j in range(1, i+1):
            A = A.at[i, j].set(A[i-1, j] + A[i-1, j-1])
    return A

def _pascal_sym(n):
    A = jnp.empty((n, n), dtype=jnp.int32)
    A = A.at[0, :].set(1)
    A = A.at[:, 0].set(1)
    for i in range(1, n):
        for j in range(1, n):
            A = A.at[i, j].set(A[i-1, j] + A[i, j-1])
    return A

def pascal(n, symmetric=False):
    """Returns a pascal matrix of size n \times n
    """
    if symmetric:
        return _pascal_sym(n)
    else:
        return _pascal_lower(n)

pascal_jit = jax.jit(pascal, static_argnums=(0, 1))
