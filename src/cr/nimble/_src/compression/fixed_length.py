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
Fixed Length Encoding of arrays
"""

from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import numpy as np


def encode_uint_arr_fl(input_arr, bits_per_sample: int):
    """Encodes an array of unsigned integers to a bit array using a fixed number of bits per sample
    """
    a = bitarray()
    max_val = (1 << bits_per_sample) - 1
    # make sure that the values are clipped
    input_arr = np.clip(input_arr, 0, max_val)
    for value in input_arr:
        value = int(value)
        a.extend(int2ba(value, bits_per_sample))
    return a

def decode_uint_arr_fl(input_bit_arr : bitarray, bits_per_sample: int):
    """Decodes an array of unsigned integers from a bit array using a fixed number of bits per sample
    """
    a = input_bit_arr
    # number of bits
    nbits = len(a)
    # number of samples
    n = nbits // bits_per_sample
    output = np.empty(n, dtype=np.int)
    idx = 0
    for i in range(n):
        # read the value
        value = ba2int(a[idx:idx+bits_per_sample])
        idx += bits_per_sample
        output[i] = value
    return output


def encode_int_arr_sgn_mag_fl(input_arr, bits_per_sample: int):
    """Encodes an array of integers to a bit array using a sign bit and a fixed number of bits per sample for magnitude
    """
    a = bitarray()
    max_val = (1 << bits_per_sample) - 1
    for value in input_arr:
        value = int(value)
        sign, value = (0, value) if value >= 0 else (1, -value)
        # make sure that the values are clipped
        value = max_val if value > max_val else value
        a.append(sign)
        a.extend(int2ba(value, bits_per_sample))
    return a


def decode_int_arr_sgn_mag_fl(input_bit_arr : bitarray, bits_per_sample: int):
    """Decodes an array of integers from a bit array using a sign bit and a fixed number of bits per sample for magnitude
    """
    a = input_bit_arr
    # number of bits
    nbits = len(a)
    # number of samples
    n = nbits // (bits_per_sample + 1)
    output = np.empty(n, dtype=np.int)
    idx = 0
    for i in range(n):
        # read the sign bit
        s = a[idx]
        idx += 1
        # read the value
        value = ba2int(a[idx:idx+bits_per_sample])
        idx += bits_per_sample
        # combine sign and value
        value = -value if s else value
        output[i] = value
    return output

