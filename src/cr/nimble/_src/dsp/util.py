# Copyright 2022-Present CR-Suite Development Team
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

def sliding_windows_rw(x, wlen, overlap):
    """Converts a signal into sliding windows (per row) with the specified overlap
    """
    step = wlen - overlap
    starts = jnp.arange(0, len(x) - wlen + 1, step)
    block = jnp.arange(wlen)
    idx = starts[:, None] + block[None, :]
    return x[idx]

def sliding_windows_cw(x, wlen, overlap):
    """Converts a signal into sliding windows (per column) with the specified overlap
    """
    return sliding_windows_rw(x, wlen, overlap).T
