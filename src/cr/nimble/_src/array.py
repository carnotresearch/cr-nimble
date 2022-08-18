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

import jax.numpy as jnp

def hermitian(A):
    r"""Returns the Hermitian transpose of an array

    Args:
        A (jax.numpy.ndarray): An array

    Returns:
        (jax.numpy.ndarray): An array: :math:`A^H`
    """
    return jnp.conjugate(A.T)

def check_shapes_are_equal(array1, array2):
    """Raise an error if the shapes of the two arrays do not match.
    
    Raises:
        ValueError: if the shape of two arrays is not same
    """
    if not array1.shape == array2.shape:
        raise ValueError('Input arrays must have the same shape.')
    return
