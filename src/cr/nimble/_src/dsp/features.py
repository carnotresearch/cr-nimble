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


def dynamic_range(x):
    """Returns the ratio of largest and smallest values (by magnitude) in x (dB)

    Args:
        x (jax.numpy.ndarray): A signal

    Returns:
        (float): The dynamic range between largest and smallest value

    Note:
        This function is not suitable for sparse signals where some values are actually 0

    See Also:
        :func:`nonzero_dynamic_range`
    """
    x = jnp.sort(jnp.abs(x))
    return 20 * jnp.log10(x[-1] / x[0])


def nonzero_dynamic_range(x):
    """Returns the ratio of largest and smallest non-zero values (by magnitude) in x (dB)

    Args:
        x (jax.numpy.ndarray): A sparse/compressible signal

    Returns:
        (float): The dynamic range between largest and smallest nonzero value

    See Also:
        :func:`dynamic_range`
    """
    x = jnp.sort(jnp.abs(x))
    idx = jnp.argmax(x != 0)
    return 20 * jnp.log10(x[-1] / x[idx])

