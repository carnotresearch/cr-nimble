# Copyright 2021 CR-Suite Development Team
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

from jax import jit, lax
import jax.numpy as jnp

from .util import promote_arg_dtypes

def AH_v(A, v):
    r"""Returns :math:`A^H v` for a given matrix A and a vector v

    Args:
        A (jax.numpy.ndarray): A matrix
        v (jax.numpy.ndarray): A vector

    Returns:
        (jax.numpy.ndarray): A vector: :math:`A^H v`

    This is definitely faster on large matrices
    """
    return jnp.conjugate((jnp.conjugate(v.T) @ A).T)


def mat_transpose(x):
    """Returns the transpose of an array of matrices

    Args:
        x (jax.numpy.ndarray): An nd-array (2 or more dimensions)

    Returns:
        (jax.numpy.ndarray): Array with last two dimensions transposed
    """
    return jnp.swapaxes(x, -1, -2)


@jit
def mat_hermitian(a):
    """Returns the conjugate transpose of an array of matrices

    Args:
        A (jax.numpy.ndarray): A JAX array (2 or more dimensions)

    Returns:
        jax.numpy.ndarray: Conjugate transpose of the array
    """
    return jnp.conjugate(jnp.swapaxes(a, -1, -2))

@jit
def is_matrix(A):
    """Checks if an array is a matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a matrix, False otherwise.
    """
    return A.ndim == 2

@jit
def is_square(A):
    """Checks if an array is a square matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a square matrix, False otherwise.
    """
    shape = A.shape
    return A.ndim == 2 and shape[0] == shape[1]

@jit
def is_symmetric(A):
    """Checks if an array is a symmetric matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a symmetric matrix, False otherwise.
    """
    if A.ndim != 2: 
        return False
    return jnp.array_equal(A, A.T)

@jit
def is_hermitian(A):
    """Checks if an array is a Hermitian matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a Hermitian matrix, False otherwise.
    """
    shape = A.shape
    if A.ndim != 2: 
        return False
    if shape[0] != shape[1]:
        return False
    return jnp.allclose(A, mat_hermitian(A), atol=1e-6)

def is_positive_definite(A):
    """Checks if an array is a symmetric positive definite matrix

    Args:
        A (jax.numpy.ndarray): A JAX array


    Returns:
        bool: True if the array is a symmetric positive definite matrix, False otherwise.

    Symmetric positive definite matrices have real and positive eigen values.
    This function checks if all the eigen values are positive. 
    """
    if A.ndim != 2: 
        return False
    A = promote_arg_dtypes(A)
    is_sym = jnp.array_equal(A, A.T)
    # check for eigen values only if we know that the matrix is symmetric
    is_pd = lax.cond(is_sym, 
        lambda _ : jnp.all(jnp.real(jnp.linalg.eigvals(A)) > 0), 
        lambda _ : False,
        None) 
    return is_pd


@jit
def has_orthogonal_columns(A, atol=1e-6):
    """Checks if a matrix has orthogonal columns

    Args:
        A (jax.numpy.ndarray): A JAX real 2D array


    Returns:
        bool: True if the matrix has orthogonal columns, False otherwise.
    """
    G = A.T @ A
    m = G.shape[0]
    I = jnp.eye(m)
    return jnp.allclose(G, I, atol=m*m*atol)


@jit
def has_orthogonal_rows(A, atol=1e-6):
    """Checks if a matrix has orthogonal rows

    Args:
        A (jax.numpy.ndarray): A JAX real 2D array


    Returns:
        bool: True if the matrix has orthogonal rows, False otherwise.
    """
    G = A @ A.T
    m = G.shape[0]
    I = jnp.eye(m)
    return jnp.allclose(G, I, atol=m*m*atol)

@jit
def has_unitary_columns(A):
    """Checks if a matrix has unitary columns

    Args:
        A (jax.numpy.ndarray): A JAX real or complex 2D array


    Returns:
        bool: True if the matrix has unitary columns, False otherwise.
    """
    G = mat_hermitian(A) @ A
    m = G.shape[0]
    I = jnp.eye(m)
    return jnp.allclose(G, I, atol=m*1e-6)

@jit
def has_unitary_rows(A):
    """Checks if a matrix has unitary rows

    Args:
        A (jax.numpy.ndarray): A JAX real or complex 2D array


    Returns:
        bool: True if the matrix has unitary rows, False otherwise.
    """
    G = A @ mat_hermitian(A)
    m = G.shape[0]
    I = jnp.eye(m)
    return jnp.allclose(G, I, atol=m*1e-6)


def off_diagonal_elements(A):
    """Returns the off diagonal elements of a matrix A

    Args:
        A (jax.numpy.ndarray): A real 2D matrix

    Returns:
        (jax.numpy.ndarray): A vector of off-diagonal elements in A
    """
    mask = ~jnp.eye(*A.shape, dtype=bool)
    return A[mask]

def off_diagonal_min(A):
    """Returns the minimum of the off diagonal elements

    Args:
        A (jax.numpy.ndarray): A real 2D matrix

    Returns:
        (float): The smallest off-diagonal element in A
    """
    off_diagonal_entries = off_diagonal_elements(A)
    return jnp.min(off_diagonal_entries)

def off_diagonal_max(A):
    """Returns the maximum of the off diagonal elements

    Args:
        A (jax.numpy.ndarray): A real 2D matrix

    Returns:
        (float): The largest off-diagonal element in A
    """
    off_diagonal_entries = off_diagonal_elements(A)
    return jnp.max(off_diagonal_entries)

def off_diagonal_mean(A):
    """Returns the maximum of the off diagonal elements

    Args:
        A (jax.numpy.ndarray): A real 2D matrix

    Returns:
        (float): The mean of all off-diagonal elements in A
    """
    off_diagonal_entries = off_diagonal_elements(A)
    return jnp.mean(off_diagonal_entries)

@jit
def set_diagonal(A, value):
    """Sets the diagonal elements to a specific value

    Args:
        A (jax.numpy.ndarray): A 2D matrix
        value (float) : A value to be added to the diagonal elements

    Returns:
        (jax.numpy.ndarray): Matrix with updated diagonal
    """
    indices = jnp.diag_indices(A.shape[0])
    return A.at[indices].set(value)

@jit
def add_to_diagonal(A, value):
    """Add a specific value to the diagonal elements 

    Args:
        A (jax.numpy.ndarray): A 2D matrix
        value (float) : A value to be added to the diagonal elements

    Returns:
        (jax.numpy.ndarray): Matrix with updated diagonal
    """
    indices = jnp.diag_indices(A.shape[0])
    return A.at[indices].add(value)

@jit
def abs_max_idx_cw(A):
    """Returns the index of entry with highest magnitude in each column
    """
    return jnp.argmax(jnp.abs(A), axis=0)

@jit
def abs_max_idx_rw(A):
    """Returns the index of entry with highest magnitude in each row
    """
    return jnp.argmax(jnp.abs(A), axis=1)



@jit
def diag_premultiply(d, A):
    """Compute D @ A where D is a diagonal matrix with entries from vector d
    """
    return jnp.multiply(d[:, None], A)

@jit
def diag_postmultiply(A, d):
    """Compute A @ D where D is a diagonal matrix with entries from vector d
    """
    return jnp.multiply(A, d)


def block_diag(A, b):
    """Extracts the block diagonal from the given matrix 

    Args:
        A (jax.numpy.ndarray): A 2D matrix
        b (int) : The size of each block

    Returns:
        An array of diagonal blocks: 3D array of shape m x b x b
        where m is the number of blocks

    Note:
        b is a static argument
    """
    n = A.shape[0]
    nb = n // b
    starts = [i*b for i in range(nb)]
    return jnp.array([A[s:s+b,s:s+b] for s in starts])

block_diag_jit = jit(block_diag, static_argnums=(1,))


def mat_column_blocks(A, n_blocks):
    """Splits the columns of a matrix into blocks and returns a 3D array

    Args:
        A (jax.numpy.ndarray): A 2D matrix
        n_blocks (int) : The number of blocks


    Returns:
        An array of matrices where each matrix is a block of columns

    Note:
        n_blocks is a static argument.
        The number of columns in A must be a multiple of n_blocks
    """
    m, n = A.shape
    blocks = A.swapaxes(0, 1).reshape(n_blocks, -1, m).swapaxes(1,2)
    return blocks
