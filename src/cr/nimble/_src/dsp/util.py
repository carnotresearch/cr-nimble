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


def norm_freq(frequency, sampling_rate):
    """Returns the normalized frequency

    The Nyquist frequency is the half of the sampling rate.
    In normalized range, the Nyquist frequency has a value of 1.
    If sampling rate is 200 Hz and signal frequency is
    20 Hz, then Nyquist frequency is 100 Hz and the
    normalized frequency is 0.2.

    Args:
        frequency (float): Frequency in Hz.
        sampling_rate (float): Sampling rate of signal in Hz.

    Returns:
        float: Normalized sampling frequency
    """
    return 2.0 * frequency / sampling_rate
