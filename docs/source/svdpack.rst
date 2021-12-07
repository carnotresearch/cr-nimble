Singular Value Decomposition 
==================================

.. contents::
    :depth: 2
    :local:

.. currentmodule:: cr.nimble


Fundamental Subspaces
--------------------------

.. autosummary::
  :toctree: _autosummary

    orth
    row_space
    null_space
    left_null_space
    effective_rank
    effective_rank_from_svd
    singular_values


SVD for Bidiagonal Matrices
------------------------------

.. currentmodule:: cr.nimble.svd

.. autosummary::
    :toctree: _autosummary
    bdsqr
    bdsqr_jit


Truncated SVD
----------------------------------


.. autosummary::
    :toctree: _autosummary
    lansvd_simple
    lansvd_simple_jit
    lanbpro_init
    lanbpro_iteration
    lanbpro_iteration_jit
    lanbpro
    lanbpro_jit
