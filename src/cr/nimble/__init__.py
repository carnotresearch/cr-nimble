# Copyright 2021 CR-Nimble Development Team
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
Linear algebra utility functions
"""
# pylint: disable=W0611


from cr.nimble._src.util import (
    platform,
    is_cpu,
    is_gpu,
    is_tpu,
    KEY0,
    KEYS,
    promote_arg_dtypes,
    canonicalize_dtype,
    promote_to_complex,
    promote_to_real,
    integer_types,
    integer_ranges,
    dtype_ranges,
    nbytes_live_buffers,
)

from cr.nimble._src.array import (
    hermitian,
    check_shapes_are_equal
)

from cr.nimble._src.matrix import (
    AH_v,
)

from cr.nimble._src.vector import (
    is_scalar,
    is_vec,
    is_line_vec,
    is_row_vec,
    is_col_vec,
    to_row_vec,
    to_col_vec,
    vec_unit,
    vec_unit_jit,
    vec_shift_right,
    vec_rotate_right,
    vec_shift_left,
    vec_rotate_left,
    vec_shift_right_n,
    vec_rotate_right_n,
    vec_shift_left_n, 
    vec_rotate_left_n,
    vec_safe_divide_by_scalar,   
)

from cr.nimble._src.norm import (
    norm_l1,
    sqr_norm_l2,
    norm_l2,
    norm_linf,

    norms_l1_cw,
    norms_l1_rw,
    norms_l2_cw,
    norms_l2_rw,
    norms_linf_cw,
    norms_linf_rw,
    sqr_norms_l2_cw,
    sqr_norms_l2_rw,


    normalize_l1_cw,
    normalize_l1_rw,
    normalize_l2_cw,
    normalize_l2_rw,
)

from cr.nimble._src.linear import (
    point2d,
    vec2d,
    rotate2d_cw,
    rotate2d_ccw,
    reflect2d,
)

from cr.nimble._src.triangular import (
    solve_Lx_b,
    solve_LTx_b,
    solve_Ux_b,
    solve_UTx_b,
    solve_spd_chol
)

from cr.nimble._src.householder import (
    householder_vec,
    householder_matrix,
    householder_premultiply,
    householder_postmultiply,
    householder_ffm_jth_v_beta,
    householder_ffm_premultiply,
    householder_ffm_backward_accum,
    householder_ffm_to_wy,
    householder_qr_packed,
    householder_split_qf_r,
    householder_qr,
)

from cr.nimble._src.chol import (
    cholesky_update_on_add_column,
    cholesky_build_factor
)

# These functions are not JIT ready
from cr.nimble._src.householder import (
    householder_vec_
)

from cr.nimble._src.svd_utils import (
    orth,
    orth_jit,
    row_space,
    row_space_jit,
    null_space,
    null_space_jit,
    left_null_space,
    left_null_space_jit,
    effective_rank,
    effective_rank_jit,
    effective_rank_from_svd,
    singular_values
)

from cr.nimble._src.dls import (
    mult_with_submatrix,
    solve_on_submatrix
)


from cr.nimble._src.standard_matrices import (
    gaussian_mtx
)