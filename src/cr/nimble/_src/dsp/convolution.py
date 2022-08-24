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

from jax import jit
from jax.scipy import signal


def vec_convolve(x, h):
    """1D full convolution based on a hack suggested by Jake Vanderplas

    See https://github.com/google/jax/discussions/7961 for details
    """
    return signal.convolve(x[None], h[None])[0]

vec_convolve_jit = jit(vec_convolve)

