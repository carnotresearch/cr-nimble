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
from jax import jit
import jax.numpy.fft as jfft

def interpft(x, N):
    """Interpolates x to n points in Fourier Transform domain
    """
    n = len(x)
    assert n < N
    a = jfft.fft(x)
    nyqst = (n + 1) // 2
    z = jnp.zeros(N -n)
    a1 = a[:nyqst+1]
    a2 = a[nyqst+1:]
    b = jnp.concatenate((a1, z, a2))
    if n % 2 == 0:
        b = b.at[nyqst].set(b[nyqst] /2 )
        b = b.at[nyqst + N -n].set(b[nyqst])
    y = jfft.ifft(b)
    if jnp.isrealobj(x):
        y = jnp.real(y)
    # scale it up
    y = y * (N / n)
    return y

