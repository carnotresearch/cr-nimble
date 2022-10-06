"""
Utility functions for working with vectors
"""

from jax import jit, lax, vmap
import jax.numpy as jnp

from .util import promote_arg_dtypes

def is_scalar(x):
    """Returns if x is a scalar 

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a scalar quantity (i.e. ndim==0).
    """
    return x.ndim == 0

def is_vec(x):
    """Returns if x is a line vector or row vector or column vector

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a line vector or a row vector or a column vector.
    """
    return x.ndim == 1 or (x.ndim == 2 and 
        (x.shape[0] == 1 or x.shape[1] == 1))

def is_line_vec(x):
    """Returns if x is a line vector

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a line vector.
    """
    return x.ndim == 1

def is_row_vec(x):
    """Returns if x is a row vector

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a row vector.
    """
    return x.ndim == 2 and x.shape[0] == 1 

def is_col_vec(x):
    """Returns if x is a column vector

    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a column vector.
    """
    return x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1)

def is_increasing_vec(x):
    """Returns if x is a vector with (strictly) increasing values
    
    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is an increasing vector.
    """
    return jnp.all(jnp.diff(x) > 0)

def is_decreasing_vec(x):
    """Returns if x is a vector with (strictly) decreasing values
    
    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a decreasing vector.
    """
    return jnp.all(jnp.diff(x) < 0)

def is_nonincreasing_vec(x):
    """Returns if x is a vector with non-increasing values
    
    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a non-increasing vector.
    """
    return jnp.all(jnp.diff(x) <= 0)

def is_nondecreasing_vec(x):
    """Returns if x is a vector with non-decreasing values
    
    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a non-decreasing vector.
    """
    return jnp.all(jnp.diff(x) >= 0)

def has_equal_values_vec(x):
    """Returns if x is a vector with equal values
    
    Args:
        x (jax.numpy.ndarray): A JAX array.

    Returns:
        True if x is a non-decreasing vector.
    """
    return jnp.all(x == x[0])


def to_row_vec(x):
    """Converts a line vector to a row vector

    Args:
        x (jax.numpy.ndarray): A line vector (ndim == 1).

    Returns:
        jax.numpy.ndarray: A row vector.
    """
    assert x.ndim == 1
    return jnp.expand_dims(x, 0)

def to_col_vec(x):
    """Converts a line vector to a column vector

    Args:
        x (jax.numpy.ndarray): A line vector (ndim == 1).

    Returns:
        jax.numpy.ndarray: A column vector.
    """
    assert x.ndim == 1
    return jnp.expand_dims(x, 1)

def vec_unit(n, i):
    """Returns a unit vector in i-th dimension for the standard coordinate system

    Args:
        n (int): Length of the vector.
        i (int): Index/dimension of the unit vector.

    Returns:
        jax.numpy.ndarray: A line vector of length n with all zeros except a one at position i. 
    """
    return jnp.zeros(n).at[i].set(1)

vec_unit_jit = jit(vec_unit, static_argnums=(0, 1))

def vec_shift_right(x):
    """Right shift the contents of the vector

    Args:
        x (jax.numpy.ndarray): A line vector.

    Returns:
        jax.numpy.ndarray: Right shifted x. 
    """
    return jnp.zeros_like(x).at[1:].set(x[:-1])

def vec_rotate_right(x):
    """Circular right shift the contents of the vector

    Args:
        x (jax.numpy.ndarray): A line vector.

    Returns:
        jax.numpy.ndarray: Right rotated x. 
    """
    return jnp.roll(x, 1)


def vec_shift_left(x):
    """Left shift the contents of the vector

    Args:
        x (jax.numpy.ndarray): A line vector.

    Returns:
        jax.numpy.ndarray: Left shifted x. 
    """
    return jnp.zeros_like(x).at[0:-1].set(x[1:])

def vec_rotate_left(x):
    """Circular left shift the contents of the vector

    Args:
        x (jax.numpy.ndarray): A line vector.

    Returns:
        jax.numpy.ndarray: Left rotated x. 
    """
    return jnp.roll(x, -1)

def vec_shift_right_n(x, n):
    """Right shift the contents of the vector by n places

    Args:
        x (jax.numpy.ndarray): A line vector.
        n (int): Number of positions to shift.

    Returns:
        jax.numpy.ndarray: Right shifted x by n places. 
    """
    return jnp.zeros_like(x).at[n:].set(x[:-n])

def vec_rotate_right_n(x, n):
    """Circular right shift the contents of the vector by n places

    Args:
        x (jax.numpy.ndarray): A line vector.
        n (int): Number of positions to shift.

    Returns:
        jax.numpy.ndarray: Right roted x by n places. 
    """
    return jnp.roll(x, n)


def vec_shift_left_n(x, n):
    """Left shift the contents of the vector by n places

    Args:
        x (jax.numpy.ndarray): A line vector.
        n (int): Number of positions to shift.

    Returns:
        jax.numpy.ndarray: Left shifted x by n places. 
    """
    return jnp.zeros_like(x).at[0:-n].set(x[n:])

def vec_rotate_left_n(x, n):
    """Circular left shift the contents of the vector by n places

    Args:
        x (jax.numpy.ndarray): A line vector.
        n (int): Number of positions to shift.

    Returns:
        jax.numpy.ndarray: Left rotated x by n places. 
    """
    return jnp.roll(x, -n)

def vec_safe_divide_by_scalar(x, alpha):
    return lax.cond(alpha == 0, lambda x : x, lambda x: x / alpha, x)

vec_safe_divide_by_scalar_jit = jit(vec_safe_divide_by_scalar)


def vec_repeat_at_end(x, p):
    """Extends a vector by repeating it at the end (periodic extension)

    Args:
        x (jax.numpy.ndarray): A line vector.
        p (int): Number of samples by which x will be extended.

    Returns:
        jax.numpy.ndarray: x extended periodically at the end. 
    """
    n = x.shape[0]
    indices = jnp.arange(p) % n
    padding = x[indices]
    return jnp.concatenate((x, padding))

vec_repeat_at_end_jit = jit(vec_repeat_at_end, static_argnums=(1,))


def vec_repeat_at_start(x, p):
    """Extends a vector by repeating it at the start (periodic extension)

    Args:
        x (jax.numpy.ndarray): A line vector.
        p (int): Number of samples by which x will be extended.

    Returns:
        jax.numpy.ndarray: x extended periodically at the start. 
    """
    n = x.shape[0]
    indices = (jnp.arange(p) + n - p) % n
    padding = x[indices]
    return jnp.concatenate((padding, x))

vec_repeat_at_start_jit = jit(vec_repeat_at_start, static_argnums=(1,))


def vec_centered(x, length):
    """Returns the central part of a vector of a specified length

    Args:
        x (jax.numpy.ndarray): A line vector.
        length (int): Length of the central part of x which will be retained.

    Returns:
        jax.numpy.ndarray: central part of x of the specified length. 
    """
    cur_len = len(x)
    length = min(cur_len, length) 
    start = (len(x) - length) // 2
    end = start + length
    return x[start:end]

vec_centered_jit = jit(vec_centered, static_argnums=(1,))

########################################################
#  Energy
########################################################

@jit
def vec_mag_desc(a):
    """Returns the coefficients in the descending order of magnitude

    Args:
        a (jax.numpy.ndarray): A vector of coefficients
    """
    return jnp.sort(jnp.abs(a))[::-1]

@jit
def vec_to_pmf(a):
    """Computes a probability mass function from a given vector

    Args:
        a (jax.numpy.ndarray): A vector of coefficients
    """
    s = jnp.sum(a) * 1.
    return a / s

@jit
def vec_to_cmf(a):
    """Computes a cumulative mass function from a given vector

    Args:
        a (jax.numpy.ndarray): A vector of coefficients
    """
    s = jnp.sum(a) * 1.
    # normalize
    a = a / s
    # generate the CMF
    return jnp.cumsum(a)

@jit
def cmf_find_quantile_index(a, q):
    """Returns the index of a given quantile in a CMF

    Args:
        a (jax.numpy.ndarray): A vector of coefficients
    """
    return jnp.argmax(a >= q)

def num_largest_coeffs_for_energy_percent(a, p):
    """Returns the number of largest components containing a given
    percentage of energy

    Args:
        a (jax.numpy.ndarray): A vector of coefficients
        p (float): percentage of energy
    """
    # compute energies
    a = jnp.conj(a) * a
    # sort in descending order
    a = jnp.sort(a)[::-1]
    # total energy
    s = jnp.sum(a) * 1.
    # normalize
    a = a / s
    # convert to a cmf
    cmf = jnp.cumsum(a)
    # the quantile value
    q = (p - 1e-10) / 100
    # find the index
    index =  jnp.argmax(cmf >= q)
    return index + 1


def vec_swap_entries(x, i, j):
    """Swaps two entries in a vector
    """
    xi = x[i]
    xj = x[j]
    x = x.at[i].set(xj)
    x = x.at[j].set(xi)
    return x


########################################################
#  Sliding Windows
########################################################

def vec_to_windows(x, wlen):
    """Constructs windows of a given length from the vector

    Args:
        x (jax.numpy.ndarray): A line vector
        wlen: length of each window

    Returns:
        jax.numpy.ndarray: A matrix of shape (wlen, m) where
        m is the number of windows

    Notes:
    - Drops extra samples from the end if the last window is not complete
    """
    n = len(x)
    # number of windows
    m = n // wlen
    # total samples to be kept
    s = m * wlen
    return jnp.reshape(x[:s], (m, wlen)).T

vec_to_windows_jit = jit(vec_to_windows, static_argnums=(1,))


########################################################
#  Circular Buffer
########################################################

def cbuf_push_left(buf, val):
    """Left shift the contents of the vector

    Args:
        buf (jax.numpy.ndarray): A circular buffer
        val: A value to be pushed in the buffer from left

    Returns:
        jax.numpy.ndarray: modified buffer
    """
    return buf.at[1:].set(buf[:-1]).at[0].set(val)

def cbuf_push_right(buf, val):
    """Left shift the contents of the vector

    Args:
        buf (jax.numpy.ndarray): A circular buffer
        val: A value to be pushed in the buffer from left

    Returns:
        jax.numpy.ndarray: modified buffer
    """
    return buf.at[:-1].set(buf[1:]).at[-1].set(val)


########################################################
#  Heap
########################################################


def is_min_heap(x):
    """ Checks if x is a min heap
    """
    n = len(x)
    idx = jnp.arange(1, n, dtype=int)
    parents = (idx-1) // 2
    return jnp.all(x[parents] <= x[1:])

def is_max_heap(x):
    """ Checks if x is a max heap
    """
    n = len(x)
    idx = jnp.arange(1, n, dtype=int)
    parents = (idx-1) // 2
    return jnp.all(x[parents] >= x[1:])

def left_child_idx(idx):
    """Returns the index of the left child
    """
    return (idx << 1) + 1

def right_child_idx(idx):
    """Returns the index of the right child
    """
    return (idx + 1) << 1

def parent_idx(idx):
    """Returns the parent index for an index
    """
    return (idx - 1) >> 1

def build_max_heap(x):
    """Converts x into a max heap
    """
    def cond_func(state):
        x,c,p = state
        return jnp.logical_and(x[c] > x[p], c > 0)

    def body_func(state):
        x,c,p = state
        xc = x[c]
        xp = x[p]
        x = x.at[c].set(xp)
        x = x.at[p].set(xc)
        c = p
        p = (p - 1) >> 1
        return x, c, p

    def main_body(i, x):
        # parent index
        p = (i - 1) >> 1
        # heapify
        x, _, _ = lax.while_loop(cond_func, body_func, (x, i, p))
        return x

    return lax.fori_loop(1, len(x), main_body, x)



def largest_plr(x, idx):
    """Return the index of the largest value between a
    parent and its children
    """
    l = (idx << 1) + 1
    r = (idx + 1) << 1
    largest = jnp.where(x[idx] < x[l], l, idx)
    largest = jnp.where(x[largest] < x[r], r, largest)
    return largest

def heapify_subtree(x, idx):
    """Heapifies a subtree starting from a given node
    """
    n = len(x)
    n2 = n >> 1

    def body_func(state):
        x, idx, _ = state
        largest = largest_plr(x, idx)
        change = largest != idx
        x = lax.cond(change,
            lambda x: vec_swap_entries(x, largest, idx),
            lambda x: x,
            x)
        return x, largest, change

    def cond_func(state):
        x, idx, change = state
        return jnp.logical_and(idx < n2, change)

    state = x, idx, True
    state = lax.while_loop(cond_func, body_func, state)
    x, idx, change = state
    return x


def delete_top_from_max_heap(x):
    """Removes the top element from a max heap retaining its heap structure
    """
    last = x[-1]
    x = x.at[0].set(last)[:-1]
    return heapify_subtree(x, 0)


def build_max_heap2(x, w=20):
    x2 = jnp.reshape(x, (-1, w))
    x2 = vmap(build_max_heap)(x2)
    return jnp.ravel(x2)
