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
Run Length Encoding


References:

https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
"""


import numpy as np

def count_runs_values(input_arr):
    """Computes run lengths of an array of integers
    """
    # make sure that input is a numpy array
    input_arr = np.asarray(input_arr)
    n = len(input_arr)
    if n == 0:
        return (np.empty(0),np.empty(0))
    if n == 1:
        return (np.ones(1), input_arr)
    # locate the changes
    changes = input_arr[1:] != input_arr[:-1]
    changes, = np.where(changes)
    # the last position should always be recorded as a change
    changes = np.append(changes, n-1)
    values = input_arr[changes]
    changes = np.insert(changes, 0, -1)
    # run lengths can be computed now
    runs = np.diff(changes)
    return runs, values

def expand_runs_values(runs, values):
    """Decodes run lengths to form an array of integers
    """
    return np.concatenate(
        [v * np.ones(r, dtype=np.int32) 
        for r, v in zip(runs, values)])
