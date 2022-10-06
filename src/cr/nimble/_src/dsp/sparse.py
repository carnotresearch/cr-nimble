# Copyright 2021 CR.Sparse Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax
import jax.numpy as jnp
from jax import random, jit
from cr.nimble import is_matrix


def randomize_rows(key, X):
    """Randomizes the rows in X

    Args:
        key: a PRNG key used as the random key.
        X (jax.numpy.ndarray): A 2D data matrix

    Returns:
        (jax.numpy.ndarray): The data matrix with randomized rows
    """
    assert is_matrix(X)
    m, n = X.shape
    r = random.permutation(key, m)
    return X[r, :]

def randomize_cols(key, X):
    """Randomizes the columns in X

    Args:
        key: a PRNG key used as the random key.
        X (jax.numpy.ndarray): A 2D data matrix

    Returns:
        (jax.numpy.ndarray): The data matrix with randomized columns
    """
    assert is_matrix(X)
    m, n = X.shape
    r = random.permutation(key, n)
    return X[:, r]



def largest_indices(x, K):
    """Returns the indices of K largest entries in x by magnitude

    Args:
        x (jax.numpy.ndarray): An data vector/point
        K (int): The number of largest entries to be identified in x

    Returns:
        (jax.numpy.ndarray): An index vector of size K identifying the K largest entries in x
        in descending order
    """
    indices = jnp.argsort(jnp.abs(x))
    return indices[:-K-1:-1]

def largest_indices_rw(X, K):
    """Returns the indices of K largest entries by magnitude in each row of X

    Args:
        X (jax.numpy.ndarray): An (S,N) data matrix with data points in rows
        K (int): The number of largest entries to be identified in each row of X

    Returns:
        (jax.numpy.ndarray): An (S,K) index matrix indices of K largest elements in each row of X
    """
    indices = jnp.argsort(jnp.abs(X), axis=1)
    return indices[:, :-K-1:-1]

def largest_indices_cw(X, K):
    """Returns the indices of K largest entries by magnitude in each column of X

    Args:
        X (jax.numpy.ndarray): An (N,S) data matrix with data points in columns
        K (int): The number of largest entries to be identified in each column of X

    Returns:
        (jax.numpy.ndarray): An (K,S) index matrix indices of K largest elements in each column of X
    """
    indices = jnp.argsort(jnp.abs(X), axis=0)
    return indices[:-K-1:-1, :]

def take_along_rows(X, indices):
    """Picks K entries from each row of X specified by indices matrix

    Args:
        X (jax.numpy.ndarray): An (S,N) data matrix with data points in rows
        indices (jax.numpy.ndarray): An (S,K) index matrix identifying the values to be picked up from X

    Returns:
        (jax.numpy.ndarray): An (S,K) data matrix subset of X containing K elements from each row of X
    """
    return jnp.take_along_axis(X, indices, axis=1)

def take_along_cols(X, indices):
    """Picks K entries from each column of X specified by indices matrix

    Args:
        X (jax.numpy.ndarray): An (N,S) data matrix with data points in columns
        indices (jax.numpy.ndarray): An (K,S) index matrix identifying the values to be picked up from X

    Returns:
        (jax.numpy.ndarray): An (K,S) data matrix subset of X containing K elements from each column of X
    """
    return jnp.take_along_axis(X, indices, axis=0)


def sparse_approximation(x, K):
    """Keeps only largest K non-zero entries by magnitude in a vector x

    Args:
        x (jax.numpy.ndarray): An data vector/point
        K (int): The number of largest entries to be kept in x

    Returns:
        (jax.numpy.ndarray): x modified so that all entries except the K largest entries are set to 0
    """
    if K == 0:
        return x.at[:].set(0)
    indices = jnp.argsort(jnp.abs(x))
    #print(x, K, indices)
    return x.at[indices[:-K]].set(0)
    
def sparse_approximation_cw(X, K):
    #return jax.vmap(sparse_approximation, in_axes=(1, None), out_axes=1)(X, K)
    """Keeps only largest K non-zero entries by magnitude in each column of X

    Args:
        X (jax.numpy.ndarray): An (N,S) data matrix with data points in columns
        K (int): The number of largest entries to be kept in each column of X

    Returns:
        (jax.numpy.ndarray): X modified so that all entries except the K largest entries are set to 0 in each column
    """
    if K == 0:
        return X.at[:].set(0)
    indices = jnp.argsort(jnp.abs(X), axis=0)
    for c in range(X.shape[1]):
        ind = indices[:-K, c]
        X = X.at[ind, c].set(0)
    return X

def sparse_approximation_rw(X, K):
    """Keeps only largest K non-zero entries by magnitude in each row of X

    Args:
        X (jax.numpy.ndarray): An (S,N) data matrix with data points in rows
        K (int): The number of largest entries to be kept in each row of X

    Returns:
        (jax.numpy.ndarray): X modified so that all entries except the K largest entries are set to 0 in each row
    """
    if K == 0:
        return X.at[:].set(0)
    indices = jnp.argsort(jnp.abs(X), axis=1)
    for r in range(X.shape[0]):
        ind = indices[r, :-K]
        X = X.at[r, ind].set(0)
    return X

def build_signal_from_indices_and_values(length, indices, values):
    """Builds a sparse signal from its non-zero entries (specified by their indices and values)

    Args:
        length (int): Length of the sparse signal
        indices (jax.numpy.ndarray): An index vector of length K identifying non-zero entries
        values (jax.numpy.ndarray): Values to be stored in the non-zero positions

    Returns:
        (jax.numpy.ndarray): Resulting sparse signal such that x[indices] == values 
    """
    x = jnp.zeros(length, dtype=values.dtype)
    indices = jnp.asarray(indices)
    values = jnp.asarray(values)
    return x.at[indices].set(values)


def nonzero_values(x):
    """Returns the values of non-zero entries in x

    Args:
        x (jax.numpy.ndarray): A sparse signal

    Returns:
        (jax.numpy.ndarray): The signal stripped of its zero values


    Note:
        This function cannot be JIT compiled as the size of output is data dependent.
    """
    return x[x != 0]

def nonzero_indices(x):
    """Returns the indices of non-zero entries in x

    Args:
        x (jax.numpy.ndarray): A sparse signal

    Returns:
        (jax.numpy.ndarray): The indices of nonzero entries in x

    Note:
        This function cannot be JIT compiled as the size of output is data dependent.

    See Also:
        :func:`support`
    """
    return jnp.nonzero(x)[0]


def support(x):
    """Returns the indices of non-zero entries in x

    Args:
        x (jax.numpy.ndarray): A sparse signal

    Returns:
        (jax.numpy.ndarray): The support of x a.k.a. the indices of nonzero entries in x
        
    Note:
        This function cannot be JIT compiled as the size of output is data dependent.

    See Also:
        :func:`nonzero_indices`
    """
    return jnp.nonzero(x)[0]



