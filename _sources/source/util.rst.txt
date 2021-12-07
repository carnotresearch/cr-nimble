Utilities in cr.nimble module
==============================

.. contents::
    :depth: 2
    :local:


.. currentmodule:: cr.nimble

Array data type utilities
-----------------------------------

.. autosummary::
  :toctree: _autosummary

  promote_arg_dtypes
  check_shapes_are_equal


Basic operations
-----------------------------------

.. autosummary::
  :toctree: _autosummary

  hermitian
  AH_v



Utilities for vectors
------------------------------------------

.. autosummary::
  :toctree: _autosummary

  is_scalar
  is_vec
  is_line_vec
  is_row_vec
  is_col_vec
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

Row wise and column wise norms for matrices
----------------------------------------------------------------------

.. autosummary::
  :toctree: _autosummary

    norms_l1_cw
    norms_l1_rw
    norms_l2_cw
    norms_l2_rw
    norms_linf_cw
    norms_linf_rw
    sqr_norms_l2_cw
    sqr_norms_l2_rw
    normalize_l1_cw
    normalize_l1_rw
    normalize_l2_cw
    normalize_l2_rw
