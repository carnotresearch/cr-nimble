# Copyright 2022 CR-Suite Development Team
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

from jax import jit
import jax.numpy as jnp
import jax.numpy.fft as jfft

def toeplitz_mat(c, r):
    """Constructs a Toeplitz matrix


    """
    c = jnp.asarray(c)
    r = jnp.asarray(r)
    m = len(c)
    n = len(r)
    # assert c[0] == r[0]
    w = jnp.concatenate((c[::-1], r[1:]))
    # backwards indices
    a = -jnp.arange(m, dtype=int)
    # print(a)
    # forwards indices
    b = jnp.arange(m-1,m+n-1, dtype=int)
    # print(b)
    # combine indices for the toeplitz matrix
    indices = a[:, None] + b[None, :]
    # print(indices)
    # form the toeplitz matrix
    mat = w[indices]
    return mat


def toeplitz_mult(w, x):
    """Multiplies a Toeplitz matrix with a vector

    Note:
        Only real matrices and vectors are supported
    """
    c, r = w
    m = len(c)
    n = len(r)
    p = m + n - 1
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ww = jnp.concatenate((c, r[-1:0:-1]))
    wf = jfft.rfft(ww).reshape(-1, 1)
    xf = jfft.rfft(x, n=p, axis=0)
    yf = wf * xf
    y = jfft.irfft(yf, n=p, axis=0)
    # drop extra values
    y = y[:m, :]
    # drop extra dimension if required
    return jnp.squeeze(y)

def circulant_mat(c):
    """Constructs a circulant matrix
    """
    # make sure that the array is flattened
    c = jnp.asarray(c).ravel()
    m = len(c)
    # extend c for the toeplitz structure
    cc = jnp.concatenate((c[::-1], c[:0:-1]))
    # backwards indices
    a = -jnp.arange(m, dtype=int)
    # forwards indices
    b = jnp.arange(m-1,m+m-1, dtype=int)
    # combine indices for the toeplitz matrix
    indices = a[:, None] + b[None, :]
    # form the circulant matrix
    mat = cc[indices]
    return mat



def circulant_mult(c, x):
    """Multiplies a circulant matrix with a vector

    Note:
        Only real matrices and vectors are supported
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    # make sure that the array is flattened
    c = jnp.asarray(c).ravel()
    m = len(c)
    cf = jfft.rfft(c).reshape(-1, 1)
    xf = jfft.rfft(x, n=m, axis=0)
    yf = xf * cf
    y = jfft.irfft(yf, n=m, axis=0)
    # drop extra dimension if required
    return jnp.squeeze(y)

