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


import struct
from bitarray import bitarray
from bitarray.util import int2ba, ba2int


def float_to_int(value):
    """Constructs an integer representation of a floating point number
    """
    s = struct.pack('>f', value)
    return struct.unpack('>l', s)[0]


def int_to_float(rep):
    """Constructs a floating point value from an integer representation
    """
    s = struct.pack('>l', rep)
    return struct.unpack('>f', s)[0]


def int_to_bitarray(value, len_bits=5):
    n = value.bit_length()
    ba = int2ba(value, length=n+1, signed=True)
    # some-bits to encode the length of integer value
    output = int2ba(n+1, length=len_bits)
    # now add the integer bit array
    output.extend(ba)
    return output

def read_int_from_bitarray(a: bitarray, pos:int, len_bits:int =5):
    # read the six bit prefix
    e = pos + len_bits
    prefix = a[pos:e]
    n = ba2int(prefix)
    pos = e
    e = e + n
    suffix = a[pos:e]
    value = ba2int(suffix, signed=True)
    return value, e


def float_to_bitarray(value):
    s = struct.pack('>f', value)
    ba = bitarray()
    ba.frombytes(s)
    return ba

def bitarray_to_float(a: bitarray):
    bytes = a.tobytes()
    value = struct.unpack('>f', bytes)[0]
    return value    

def read_float_from_bitarray(a: bitarray, pos: int):
    e = pos+32
    value = a[pos:e]
    return bitarray_to_float(value), e

