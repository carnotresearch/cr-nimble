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

"""
Signal Processing Utilities
"""

# pylint: disable=W0611


from cr.nimble._src.dsp.util import (
    sliding_windows_rw,
    sliding_windows_cw
)

from cr.nimble._src.dsp.convolution import (
    # convolution
    vec_convolve,
    vec_convolve_jit,
)

# Energy
from cr.nimble._src.dsp.energy import (

    # energy of a signal
    energy,
    find_first_signal_with_energy_le_rw,
    find_first_signal_with_energy_le_cw,
)

# Thresholding
from cr.nimble._src.dsp.thresholding import (

    hard_threshold,
    hard_threshold_sorted,
    hard_threshold_by,
    largest_indices_by,
    energy_threshold,
)

# Scaling
from cr.nimble._src.dsp.scaling import (

    scale_to_0_1,
    descale_from_0_1,
    # statistical normalization of data
    scale_0_mean_1_var,
    scale_0_mean_1_var_jit,
)

# Quantization
from cr.nimble._src.dsp.quantization import (
    quantize_1,
    inv_quantize_1,
)


# Spectrum
from cr.nimble._src.dsp.spectrum import (
    norm_freq,
    frequency_spectrum,
    power_spectrum
)

# Interpolation
from cr.nimble._src.dsp.interpolation import (
    # interpolate via fourier transform
    interpft,
)

# Signal Features
from cr.nimble._src.dsp.features import (
    dynamic_range,
    nonzero_dynamic_range,
)

# Sparse Signals
from cr.nimble._src.dsp.sparse import (
    nonzero_values,
    nonzero_indices,
    support,
    largest_indices,
    sparse_approximation,
    build_signal_from_indices_and_values,
)


# Sparse Signal Matrices
from cr.nimble._src.dsp.sparse import (
    randomize_rows,
    randomize_cols,
    # row wise
    take_along_rows,
    largest_indices_rw,
    sparse_approximation_rw,
    # column wise
    take_along_cols,
    largest_indices_cw,
    sparse_approximation_cw,
)

# Signal Comparison
from cr.nimble._src.signalcomparison import (
    SignalsComparison,
    snrs_cw,
    snrs_rw,
    snr
)

# Noise
from cr.nimble._src.noise import (
    awgn_at_snr_ms,
    awgn_at_snr_std,
    awgn_at_snr
)

# Discrete Cosine Transform
from cr.nimble._src.dsp.dct import (
    dct,
    idct,
    orthonormal_dct,
    orthonormal_idct
)


# Walsh Hadamard
from cr.nimble._src.dsp.wht import (
    fwht,
)

from cr.nimble._src.dsp.synthetic_signals import (
    time_values,
)
