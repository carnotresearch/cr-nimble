Utilities
==============================

.. contents::
    :depth: 2
    :local:


.. currentmodule:: cr.nimble


Data Type Management
---------------------------


.. autosummary::
    :toctree: _autosummary

    promote_arg_dtypes
    check_shapes_are_equal
    canonicalize_dtype
    promote_to_complex
    promote_to_real

System Information
-----------------------------

.. autosummary::
    :toctree: _autosummary

    platform
    is_cpu
    is_gpu
    is_tpu
    nbytes_live_buffers



2D Geometry
----------------------------

.. currentmodule:: cr.nimble

.. rubric:: Points and Vectors


.. autosummary::
    :toctree: _autosummary

    point2d
    vec2d

.. rubric:: Transformations

.. autosummary::
    :toctree: _autosummary

    rotate2d_cw
    rotate2d_ccw
    reflect2d
