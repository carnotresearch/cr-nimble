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


from cr.nimble._src.signal import (
    # energy of a signal
    energy,
    # interpolate via fourier transform
    interpft,
    # statistical normalization of data
    normalize,
    normalize_jit,
    # convolution
    vec_convolve,
    vec_convolve_jit,
)

from cr.nimble._src.signal import (

    find_first_signal_with_energy_le_rw,
    find_first_signal_with_energy_le_cw,
)

from cr.nimble._src.signal import (
    frequency_spectrum,
    power_spectrum
)


from cr.nimble._src.signalcomparison import (
    SignalsComparison,
    snrs_cw,
    snrs_rw,
    snr
)

from cr.nimble._src.noise import (
    awgn_at_snr
)


from cr.nimble._src.dsp.dct import (
    dct,
    idct,
    orthonormal_dct,
    orthonormal_idct
)


from cr.nimble._src.dsp.wht import (
    fwht,
)

from cr.nimble._src.dsp.synthetic_signals import (
    time_values,
)

from cr.nimble._src.dsp.util import (
    norm_freq,
)
