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


def quantize_1(x, n):
    """Quantizes a signal to n bits where signal values are bounded by 1

    Args:
        x (jax.numpy.ndarray): A signal to be quantized
        n (int): number of bits for quantization

    Returns:
        (jax.numpy.ndarray): Quantized signal with integer values
    """
    # scaling
    factor = 2**n-1
    x = factor * x
    # quantization
    x = jnp.round(x)
    # type conversion from float to int
    x = x.astype(int)
    return x


def inv_quantize_1(x, n):
    """Inverse quantizes a signal from n bits

    Args:
        x (jax.numpy.ndarray): A signal to be inverse quantized
        n (int): number of bits for quantization

    Returns:
        (jax.numpy.ndarray): Quantized signal with integer values
    """
    # scaling
    factor = 2**n-1
    x = x / factor
    return x
