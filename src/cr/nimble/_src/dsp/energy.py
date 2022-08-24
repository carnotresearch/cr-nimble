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

from cr.nimble import (
    is_matrix,
    sqr_norms_l2_rw,
    sqr_norms_l2_cw)

def find_first_signal_with_energy_le_rw(X, energy):
    """Returns the index of the first row which has energy less than the specified threshold
    """
    assert is_matrix(X)
    energies = sqr_norms_l2_rw(X)
    index = jnp.argmax(energies <= energy)
    return index if energies[index] <= energy else jnp.array(-1)

def find_first_signal_with_energy_le_cw(X, energy):
    """Returns the index of the first column which has energy less than the specified threshold
    """
    assert is_matrix(X)
    energies = sqr_norms_l2_cw(X)
    index = jnp.argmax(energies <= energy)
    return index if energies[index] <= energy else jnp.array(-1)


def energy(data, axis=-1):
    """
    Computes the energy of the signal along the specified axis
    """
    power = jnp.abs(data) ** 2
    return jnp.sum(power, axis)


