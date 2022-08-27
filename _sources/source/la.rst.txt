Linear Algebra
============================

.. contents::
    :depth: 2
    :local:

.. currentmodule:: cr.nimble



Linear Systems
------------------------

.. rubric:: Triangular Systems


.. autosummary::
  :toctree: _autosummary

    solve_Lx_b
    solve_LTx_b
    solve_Ux_b
    solve_UTx_b
    solve_spd_chol



.. rubric:: Special Dense Linear Systems


.. autosummary::
    :toctree: _autosummary

    mult_with_submatrix
    solve_on_submatrix


Singular Value Decomposition 
--------------------------------

.. rubric:: Fundamental Subspaces

.. autosummary::
  :toctree: _autosummary

    orth
    row_space
    null_space
    left_null_space
    effective_rank
    effective_rank_from_svd
    singular_values


.. rubric:: SVD for Bidiagonal Matrices

.. currentmodule:: cr.nimble.svd

.. autosummary::
    :toctree: _autosummary

    bdsqr
    bdsqr_jit


.. rubric:: Truncated SVD


.. autosummary::
    :toctree: _autosummary
    
    lansvd_simple
    lansvd_simple_jit
    lanbpro_init
    lanbpro_iteration
    lanbpro_iteration_jit
    lanbpro
    lanbpro_jit



Orthogonalization
------------------------

.. currentmodule:: cr.nimble

.. rubric:: Householder Reflections

.. autosummary::
    :toctree: _autosummary

    householder_vec
    householder_matrix
    householder_premultiply
    householder_postmultiply
    householder_ffm_jth_v_beta
    householder_ffm_premultiply
    householder_ffm_backward_accum
    householder_ffm_to_wy
    householder_qr_packed
    householder_split_qf_r
    householder_qr



Subspaces
---------------------------

.. currentmodule:: cr.nimble.subspaces

.. rubric:: Projection

.. autosummary::
    :toctree: _autosummary

    project_to_subspace
    is_in_subspace


.. rubric:: Principal Angles

.. autosummary::
    :toctree: _autosummary

    principal_angles_cos
    principal_angles_rad
    principal_angles_deg
    smallest_principal_angle_cos
    smallest_principal_angle_rad
    smallest_principal_angle_deg
    smallest_principal_angles_cos
    smallest_principal_angles_rad
    smallest_principal_angles_deg
    subspace_distance


Affine Spaces
------------------------------------

.. contents::
    :depth: 2
    :local:

.. currentmodule:: cr.nimble.affine


.. rubric:: Homogeneous Coordinate System

.. autosummary::
    :toctree: _autosummary

    homogenize
    homogenize_vec
    homogenize_cols    

Standard Matrices
------------------------------------

.. currentmodule:: cr.nimble

.. rubric:: Random matrices

.. autosummary::
    :toctree: _autosummary

    gaussian_mtx


.. rubric:: Special Matrices


.. autosummary::
    :toctree: _autosummary

    pascal
    pascal_jit
