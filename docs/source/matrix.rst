Matrices
===================

.. contents::
    :depth: 2
    :local:


.. currentmodule:: cr.nimble

Predicates
----------------------------------------------------------------------

.. autosummary::
    :toctree: _autosummary

    is_matrix
    is_square
    is_symmetric
    is_hermitian
    is_positive_definite
    has_orthogonal_columns
    has_orthogonal_rows
    has_unitary_columns
    has_unitary_rows

Matrix multiplication
----------------------------

.. autosummary::
    :toctree: _autosummary

    AH_v
    mat_transpose
    mat_hermitian

Matrix parts
------------------------

.. autosummary::
    :toctree: _autosummary

    off_diagonal_elements
    off_diagonal_min
    off_diagonal_max
    off_diagonal_mean



Row wise and column wise norms
-----------------------------------

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


Pairwise Distances
-------------------------

.. autosummary::
  :toctree: _autosummary

  pairwise_sqr_l2_distances_rw
  pairwise_sqr_l2_distances_cw
  pairwise_l2_distances_rw
  pairwise_l2_distances_cw
  pdist_sqr_l2_rw
  pdist_sqr_l2_cw
  pdist_l2_rw
  pdist_l2_cw
  pairwise_l1_distances_rw
  pairwise_l1_distances_cw
  pdist_l1_rw
  pdist_l1_cw
  pairwise_linf_distances_rw
  pairwise_linf_distances_cw
  pdist_linf_rw
  pdist_linf_cw


Sparse Matrices
-------------------------

Following functions analyze, transform, or construct representation matrices
which are known to be sparse.

