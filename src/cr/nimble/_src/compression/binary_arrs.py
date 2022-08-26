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
Run Length Encoding of Binary Maps


References:

https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
"""

from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import numpy as np

def count_binary_runs(input_arr):
    """Returns the runs of 0s and 1s in a binary map
    """
    # make sure that it is a numpy array
    input_arr = np.asarray(input_arr)
    # the first bit
    b = input_arr[0]
    # the last bit
    e = input_arr[-1]
    # extend the array
    extended = np.hstack(([1-b], input_arr, [1 - e]))
    # locate the changes
    diffs = np.diff(extended)
    markers, = np.where(diffs)
    runs = np.diff(markers)
    return runs

B00 = bitarray('00')
B01 = bitarray('01')
B10 = bitarray('10')
B11 = bitarray('11')
NUM_BITS_RUN_LEN = 4


def encode_binary_arr(input_arr):
    """Encodes a binary array into a bit array via run length encoding
    """
    # the first bit
    b = input_arr[0]
    # the runs
    runs = count_binary_runs(input_arr)
    # build the bit array
    a = bitarray()
    a.append(b)
    for run in runs:
        run = int(run)
        if run == 1:
            a.extend(B00)
            continue
        if run == 2:
            a.extend(B01)
            continue
        if run == 3:
            a.extend(B10)
            continue
        # run is 4 or more
        a.extend(B11)
        # now record number of bits for the run
        bl = run.bit_length()
        a.extend(int2ba(bl, NUM_BITS_RUN_LEN))
        # now record the run
        a.extend(int2ba(run))
    return a



def decode_binary_arr(input_bit_arr : bitarray):
    """Decodes a binary array from a bit array via run length decoding
    """
    a = input_bit_arr
    result = []
    # The first bit
    b = a[0]
    idx = 1
    # number of bits in the encoded bit array
    n = len(a)
    while idx < n:
        # read the next 2 bits
        code = a[idx:idx+2]
        idx += 2
        code = ba2int(code)
        run = code + 1
        if code == 3:
            # we need to decode run from the stream
            bl = ba2int(a[idx:idx+NUM_BITS_RUN_LEN])
            idx += NUM_BITS_RUN_LEN
            run = ba2int(a[idx:idx+bl])
            idx += bl
        for i in range(run):
            result.append(b)
        b = 1 - b
    return np.array(result)


def binary_compression_ratio(input_arr, output_arr, bits_per_sample=1):
    """Returns the compression ratio of binary array compression algorithm
    """
    out_len = output_arr.nbytes * 8
    ratio = len(input_arr) * bits_per_sample / out_len
    return ratio

def binary_space_saving_ratio(input_arr, output_arr, bits_per_sample=1):
    """Returns the space saving ratio of binary array compression algorithm
    """
    out_len = output_arr.nbytes * 8
    return 1 - out_len / (len(input_arr) * bits_per_sample)
