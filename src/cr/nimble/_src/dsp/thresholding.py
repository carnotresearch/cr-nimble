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



"""
There are slightly varying approaches to thresholding.

Thresholding operators as defined in the paper: 

.. [1] Chen, Y., Chen, K., Shi, P., Wang, Y., “Irregular seismic
       data reconstruction using a percentile-half-thresholding algorithm”,
       Journal of Geophysics and Engineering, vol. 11. 2014.       
"""


import jax
import jax.numpy as jnp
from jax import jit
from cr.nimble import promote_arg_dtypes

def hard_threshold(x, K):
    """Returns the indices and corresponding values of largest K non-zero entries in a vector x

    Args:
        x (jax.numpy.ndarray): A sparse/compressible signal
        K (int): The number of largest entries to be kept in x

    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray): A tuple comprising of:
            * The indices of K largest entries in x
            * Corresponding entries in x

    See Also:
        :func:`hard_threshold_sorted`
        :func:`hard_threshold_by`
    """
    indices = jnp.argsort(jnp.abs(x))
    I = indices[:-K-1:-1]
    x_I = x[I]
    return I, x_I

def hard_threshold_sorted(x, K):
    """Returns the sorted indices and corresponding values of largest K non-zero entries in a vector x

    Args:
        x (jax.numpy.ndarray): A sparse/compressible signal
        K (int): The number of largest entries to be kept in x

    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray): A tuple comprising of:
            * The indices of K largest entries in x sorted in ascending order
            * Corresponding entries in x

    See Also:
        :func:`hard_threshold`
    """
    # Sort entries in x by their magnitude
    indices = jnp.argsort(jnp.abs(x))
    # Pick the indices of K-largest (magnitude) entries in x (from behind)
    I = indices[:-K-1:-1]
    # Make sure that indices are sorted in ascending order
    I = jnp.sort(I)
    # Pick corresponding values
    x_I = x[I]
    return I, x_I

def hard_threshold_by(x, t):
    """
    Sets all entries in x to be zero which are less than t in magnitude

    Args:
        x (jax.numpy.ndarray): A sparse/compressible signal
        t (float): The threshold value

    Returns:
        (jax.numpy.ndarray): x modified such that all values below t are set to 0

    Note:
        This function doesn't change the length of x and can be JIT compiled

    See Also:
        :func:`hard_threshold`
    """
    valid = jnp.abs(x) >= t
    return x * valid

def largest_indices_by(x, t):
    """
    Returns the locations of all entries in x which are larger than t in magnitude

    Args:
        x (jax.numpy.ndarray): A sparse/compressible signal
        t (float): The threshold value

    Returns:
        (jax.numpy.ndarray): An index vector of all entries in x which are above the threshold

    Note:
        This function cannot be JIT compiled as the length of output is data dependent

    See Also:
        :func:`hard_threshold_by`
    """
    return jnp.where(jnp.abs(x) >= t)[0]


def energy_threshold(signal, fraction):
    """
    Keeps only as much coefficients in signal so as to capture a fraction of signal energy 

    Args:
        x (jax.numpy.ndarray): A signal
        fraction (float): The fraction of energy to be preserved

    Returns:
        (jax.numpy.ndarray, jax.numpy.ndarray): A tuple comprising of:
            * Signal after thresholding
            * A binary mask of the indices to be kept

    Note:
        This function doesn't change the length of signal and can be JIT compiled

    See Also:
        :func:`hard_threshold`
    """
    # signal length
    n = signal.size
    # compute energies
    energies = signal ** 2
    # sort in descending order
    idx = jnp.argsort(energies)[::-1]
    energies = energies[idx]
    # total energy
    s = jnp.sum(energies) * 1.
    # normalize
    energies = energies / s
    # convert to a cmf
    cmf = jnp.cumsum(energies)
    # find the index
    index =  jnp.argmax(cmf >= fraction)
    # build the mask
    idx2 = jnp.arange(n)
    mask = jnp.where(idx2 <= index, 1, 0)
    # reshuffle the mask
    mask = mask.at[idx].set(mask)
    signal = signal * mask
    return signal, mask


#############################################################################
#
# Thresholding operators as defined in the paper: 
#
# .. [1] Chen, Y., Chen, K., Shi, P., Wang, Y., “Irregular seismic
#       data reconstruction using a percentile-half-thresholding algorithm”,
#       Journal of Geophysics and Engineering, vol. 11. 2014.       
#
#############################################################################


def hard_threshold_tau(x, tau):
    """Hard thresholding as per :cite:`chen2014irregular`
    """
    x = promote_arg_dtypes(x)
    # The terms that will remain non-zero after thresholding
    gamma = jnp.sqrt(2*tau)
    nonzero = jnp.abs(x) > gamma
    return nonzero * x

def soft_threshold_tau(x, tau):
    """Soft thresholding as per :cite:`chen2014irregular`
    """
    x = promote_arg_dtypes(x)
    if jnp.iscomplexobj(x):
        return jnp.maximum(jnp.abs(x) - tau, 0.) * jnp.exp(1j * jnp.angle(x))
    else:
        return jnp.maximum(0, x - tau) + jnp.minimum(0, x + tau)

def half_threshold_tau(x, tau):
    r"""Half thresholding as per :cite:`chen2014irregular`
    """
    x = promote_arg_dtypes(x)
    gamma = (54 ** (1. / 3.) / 4.) * tau ** (2. / 3.)
    nonzero = jnp.abs(x) >= gamma
    # the arc-cos term Eq 10 from paper
    phi = 2. / 3. * jnp.arccos((tau / 8.) * (jnp.abs(x) / 3.) ** (-1.5))
    # the half thresholded values for terms above gamma Eq 10
    x = 2./3. * x * (1 + jnp.cos(2. * jnp.pi / 3. - phi))
    # combine zero and non-zero terms
    return jnp.where(nonzero, x, jnp.zeros_like(x))


def hard_threshold_percentile(x, perc):
    """Percentile hard thresholding as per :cite:`chen2014irregular`
    """
    x = promote_arg_dtypes(x)
    # desired gamma
    gamma = jnp.percentile(jnp.abs(x), perc)
    # convert gamma to tau
    tau = 0.5 * gamma ** 2
    return hard_threshold_tau(x, tau)

def soft_threshold_percentile(x, perc):
    """Percentile soft thresholding as per :cite:`chen2014irregular`
    """
    x = promote_arg_dtypes(x)
    # desired gamma and tau are same
    tau = jnp.percentile(jnp.abs(x), perc)
    return soft_threshold_tau(x, tau)

def half_threshold_percentile(x, perc):
    """Percentile half thresholding as per :cite:`chen2014irregular`
    """
    x = promote_arg_dtypes(x)
    gamma = jnp.percentile(jnp.abs(x), perc)
    # convert gamma to tau
    tau = (4. / 54 ** (1. / 3.) * gamma) ** 1.5
    return half_threshold_tau(x, tau)

def gamma_to_tau_half_threshold(gamma):
    """Converts gamma to tau for half thresholding as per :cite:`chen2014irregular`
    """
    return (4. / 54 ** (1. / 3.) * gamma) ** 1.5

def gamma_to_tau_hard_threshold(gamma):
    """Converts gamma to tau for hard thresholding as per :cite:`chen2014irregular`
    """
    return  0.5 * gamma ** 2
