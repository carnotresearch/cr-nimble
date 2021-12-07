# Copyright 2021 CR-Nimble Development Team
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

