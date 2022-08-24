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

from cr.nimble import next_pow_of_2

def norm_freq(frequency, sampling_rate):
    """Returns the normalized frequency

    The Nyquist frequency is the half of the sampling rate.
    In normalized range, the Nyquist frequency has a value of 1.
    If sampling rate is 200 Hz and signal frequency is
    20 Hz, then Nyquist frequency is 100 Hz and the
    normalized frequency is 0.2.

    Args:
        frequency (float): Frequency in Hz.
        sampling_rate (float): Sampling rate of signal in Hz.

    Returns:
        float: Normalized sampling frequency
    """
    return 2.0 * frequency / sampling_rate


def frequency_spectrum(x, dt=1.):
    """Frequency spectrum of 1D data using FFT
    """
    n = len(x)
    nn = next_pow_of_2(n)
    X = jfft.fft(x, nn)
    f = jfft.fftfreq(nn, d=dt)
    X = jfft.fftshift(X)
    f = jfft.fftshift(f)
    return f, X

def power_spectrum(x, dt=1.):
    """Power spectrum of 1D data using FFT
    """
    n = len(x)
    T = dt * n
    f, X = frequency_spectrum(x, dt)
    nn = len(f)
    n2 = nn // 2
    f = f[n2:]
    X = X[n2:]
    sxx = (X * jnp.conj(X)) / T
    sxx = jnp.abs(sxx)
    return f, sxx
