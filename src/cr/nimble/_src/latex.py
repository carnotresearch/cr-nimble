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


import numpy as np

def to_tex_matrix(a, env='bmatrix'):
    a = np.asarray(a)
    assert a.ndim == 2, 'Input must be a matrix'
    # use numpy string conversion first
    text = str(a)
    # remove the brackets
    text = text.replace('[', '')
    text = text.replace(']', '')
    # split the text into lines
    lines = text.splitlines()
    # fill in the ampersands
    lines = [' & '.join(line.split()) for line in lines]
    # combine the lines
    body = '\n'.join([line + '\\\\' for line in lines])
    # add the env block
    body = f'\\begin{{{env}}}\n{body}\n\\end{{{env}}}'
    return body
