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
    is_increasing_vec
    is_decreasing_vec
    is_nonincreasing_vec
    is_nondecreasing_vec
    has_equal_values_vec

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
    vec_repeat_at_end
    vec_repeat_at_start
    vec_centered
    vec_unit_jit
    vec_repeat_at_end_jit
    vec_repeat_at_start_jit
    vec_centered_jit
    vec_swap_entries

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


Circular Buffer
-------------------------


.. autosummary::
    :toctree: _autosummary

    cbuf_push_left
    cbuf_push_right


Binary Heap
-------------------------


.. autosummary::
    :toctree: _autosummary

    is_min_heap
    is_max_heap
    left_child_idx
    right_child_idx
    parent_idx
    build_max_heap
    largest_plr
    heapify_subtree
    delete_top_from_max_heap


Miscellaneous
-------------------------


.. autosummary::
    :toctree: _autosummary

    vec_mag_desc
    vec_to_pmf
    vec_to_cmf
    cmf_find_quantile_index
    num_largest_coeffs_for_energy_percent