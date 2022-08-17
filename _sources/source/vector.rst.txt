Vectors
========================

.. contents::
    :depth: 2
    :local:


Predicates
-------------------

.. currentmodule:: cr.nimble

.. autosummary::
    :toctree: _autosummary

    is_scalar
    is_vec
    is_line_vec
    is_row_vec
    is_col_vec

Unary Operations
----------------------------------

.. autosummary::
    :toctree: _autosummary

    to_row_vec
    to_col_vec
    vec_unit
    vec_shift_right
    vec_rotate_right
    vec_shift_left
    vec_rotate_left
    vec_shift_right_n
    vec_rotate_right_n
    vec_shift_left_n
    vec_rotate_left_n 
    vec_unit_jit
    vec_repeat_at_end
    vec_repeat_at_start
    vec_centered
    vec_unit_jit
    vec_repeat_at_end_jit
    vec_repeat_at_start_jit
    vec_centered_jit

Norm
----------------------------------

.. autosummary::
    :toctree: _autosummary

    norm_l1
    norm_l2
    norm_linf
    sqr_norm_l2
    normalize_l1
    normalize_l2
    normalize_linf



Sparse Vectors
------------------------------------

Following functions analyze, transform, or construct vectors
which are known to be sparse.

.. autosummary::
  :toctree: _autosummary

    nonzero_values
    nonzero_indices
    support
    largest_indices
    largest_indices_by
    hard_threshold
    hard_threshold_sorted
    hard_threshold_by
    sparse_approximation
    build_signal_from_indices_and_values
    dynamic_range
    nonzero_dynamic_range

Miscellaneous
-------------------------


.. autosummary::
    :toctree: _autosummary

    vec_mag_desc
    vec_to_pmf
    vec_to_cmf
    cmf_find_quantile_index
    num_largest_coeffs_for_energy_percent