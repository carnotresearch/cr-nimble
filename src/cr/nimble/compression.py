# Copyright 2022 CR-Suite Development Team
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
Basic data compression routines
"""
# pylint: disable=W0611


from cr.nimble._src.compression.binary_arrs import (
    count_binary_runs,
    encode_binary_arr,
    decode_binary_arr,
    binary_compression_ratio,
    binary_space_saving_ratio,
)

from cr.nimble._src.compression.fixed_length import (
    encode_uint_arr_fl,
    decode_uint_arr_fl,
    encode_int_arr_sgn_mag_fl,
    decode_int_arr_sgn_mag_fl
)

from cr.nimble._src.compression.run_length import (
    count_runs_values,
    expand_runs_values
)

from cr.nimble._src.compression.bits import (
    float_to_int,
    int_to_float,
    int_to_bitarray,
    read_int_from_bitarray,
    float_to_bitarray,
    bitarray_to_float,
    read_float_from_bitarray
)
