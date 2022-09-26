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

from cr.nimble._src.discrete.number import (
    next_pow_of_2,
    is_integer,
    is_positive_integer,
    is_negative_integer,
    is_odd,
    is_even,
    is_odd_natural,
    is_even_natural,
    is_power_of_2,
    is_perfect_square,
    integer_factors_close_to_sqr_root
)

from cr.nimble._src.array import (
    hermitian,
    check_shapes_are_equal
)

from cr.nimble._src.vector import (
    is_scalar,
    is_vec,
    is_line_vec,
    is_increasing_vec,
    is_decreasing_vec,
    is_nonincreasing_vec,
    is_nondecreasing_vec,
    has_equal_values_vec,
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
    vec_repeat_at_end,
    vec_repeat_at_end_jit,
    vec_repeat_at_start,
    vec_repeat_at_start_jit,
    vec_centered,
    vec_centered_jit,
    vec_to_windows,
    vec_to_windows_jit,
    vec_mag_desc,
    vec_swap_entries,
    vec_to_pmf,
    vec_to_cmf,
    cmf_find_quantile_index,
    num_largest_coeffs_for_energy_percent
)

from cr.nimble._src.vector import (
    is_min_heap,
    is_max_heap,
    left_child_idx,
    right_child_idx,
    parent_idx,
    build_max_heap,
    largest_plr,
    heapify_subtree,
    delete_top_from_max_heap
)

from cr.nimble._src.vector import (
    cbuf_push_left,
    cbuf_push_right
)

from cr.nimble._src.matrix import (
    AH_v,
    mat_transpose,
    mat_hermitian,
    is_matrix,
    is_square,
    is_symmetric,
    is_hermitian,
    is_positive_definite,
    has_orthogonal_columns,
    has_orthogonal_rows,
    has_unitary_columns,
    has_unitary_rows,
    off_diagonal_elements,
    off_diagonal_min,
    off_diagonal_max,
    off_diagonal_mean,
    set_diagonal,
    add_to_diagonal,
    abs_max_idx_cw,
    abs_max_idx_rw,
    diag_premultiply,
    diag_postmultiply,
    block_diag,
    block_diag_jit,
    mat_column_blocks
)



from cr.nimble._src.norm import (
    norm_l1,
    sqr_norm_l2,
    norm_l2,
    norm_linf,

    normalize_l1,
    normalize_l2,
    normalize_linf,

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


from cr.nimble._src.distance import (
    pairwise_sqr_l2_distances_rw,
    pairwise_sqr_l2_distances_cw,
    pairwise_l2_distances_rw,
    pairwise_l2_distances_cw,
    pdist_sqr_l2_rw,
    pdist_sqr_l2_cw,
    pdist_l2_rw,
    pdist_l2_cw,
    # Manhattan distances
    pairwise_l1_distances_rw,
    pairwise_l1_distances_cw,
    pdist_l1_rw,
    pdist_l1_cw,

    # Chebychev distance
    pairwise_linf_distances_rw,
    pairwise_linf_distances_cw,
    pdist_linf_rw,
    pdist_linf_cw
)


from cr.nimble._src.metrics import (
    mean_squared,
    mean_squared_error,
    root_mean_squared,
    root_mse,
    normalization_factor,
    normalized_root_mse,
    normalized_mse,
    peak_signal_noise_ratio,
    signal_noise_ratio,
    percent_rms_diff,
    prd,
    compression_ratio,
    percent_space_saving,
    prd_to_snr,
    cr_to_pss,
    pss_to_cr
)

from cr.nimble._src.ndarray import (
    arr_largest_index,
    arr_l1norm,
    arr_l2norm,
    arr_l2norm_sqr,
    arr_vdot,
    arr_rdot,
    arr_rnorm_sqr,
    arr_rnorm,
    arr2vec,
    log_pos,
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
    gaussian_mtx,
    pascal,
    pascal_jit
)

from cr.nimble._src.toeplitz import (
    toeplitz_mat,
    toeplitz_mult,
    circulant_mat,
    circulant_mult
)

########################################
# Similarity
########################################

from cr.nimble._src.similarity import (
    dist_to_gaussian_sim,
    sqr_dist_to_gaussian_sim,
    eps_neighborhood_sim
)

########################################
# From numpy array to latex
########################################


from cr.nimble._src.latex import (
    to_tex_matrix
)

########################################
# Miscellaneous stuff
########################################
