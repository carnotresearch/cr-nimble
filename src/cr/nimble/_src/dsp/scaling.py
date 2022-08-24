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


def normalize(data, axis=-1):
    """Normalizes a data vector (data - mu) / sigma 

    Args:
        data (jax.numpy.ndarray): A data vector or array
        axis (int): For nd arrays, the axis along which the data normalization will be done

    Returns:
        (jax.numpy.ndarray): Normalized data vector/array
    """
    mu = jnp.mean(data, axis)
    data = data - mu
    variance = jnp.var(data, axis)
    data = data / jnp.sqrt(variance)
    return data

normalize_jit = jit(normalize, static_argnums=(1,))


def scale_to_0_1(x):
    """Scales a signal to the range of 0 and 1

    Args:
        x (jax.numpy.ndarray): A signal to be scaled

    Returns:
        (float, float, jax.numpy.ndarray): A tuple comprising of:
            * The amount of shift
            * The scale factor
            * Scaled signal
    """
    shift = jnp.min(x)
    x = x - shift
    scale = jnp.max(x)
    x = x / scale
    return shift, scale, x
