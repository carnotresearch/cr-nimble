.. _api:dsp:

Digital Signal Processing
===============================

.. contents::
    :depth: 2
    :local:

The ``CR-Nimble`` library has some handy digital signal processing routines
implemented in JAX. They are available as part of the ``cr.nimble.dsp``
package.


.. currentmodule:: cr.nimble.dsp

Signal Energy
-------------------------------

.. autosummary::
    :toctree: _autosummary

    energy

Thresholding
-------------------------------

.. autosummary::
    :toctree: _autosummary

    hard_threshold
    hard_threshold_sorted
    hard_threshold_by
    largest_indices_by
    energy_threshold



Scaling
-------------------------------

.. autosummary::
    :toctree: _autosummary

    scale_to_0_1
    scale_0_mean_1_var

Quantization
-------------------------------

.. autosummary::
    :toctree: _autosummary

    quantize_1


Spectrum Analysis
-------------------------------

.. autosummary::
    :toctree: _autosummary

    norm_freq
    frequency_spectrum
    power_spectrum

Interpolation
-------------------------------

.. autosummary::
  :toctree: _autosummary

  interpft



Sparse Signals
------------------------------------

Following functions analyze, transform, or construct signals
which are known to be sparse.

.. autosummary::
    :toctree: _autosummary

    nonzero_values
    nonzero_indices
    support
    largest_indices
    sparse_approximation
    build_signal_from_indices_and_values


Matrices of Sparse Signals
------------------------------------

Following functions analyze, transform, or construct
collections of sparse signals organized as matrices.

.. autosummary::
    :toctree: _autosummary

    randomize_rows
    randomize_cols

.. rubric:: Sparse representation matrices (row-wise)

.. autosummary::
    :toctree: _autosummary

    take_along_rows
    largest_indices_rw
    sparse_approximation_rw

.. rubric:: Sparse representation matrices (column-wise)

.. autosummary::
    :toctree: _autosummary

    take_along_cols
    largest_indices_cw
    sparse_approximation_cw





Artificial Noise
-----------------------------------


.. autosummary::
  :toctree: _autosummary

  awgn_at_snr_ms
  awgn_at_snr_std


Synthetic Signals
-----------------------

.. currentmodule:: cr.nimble.dsp.signals

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    pulse
    transient_sine_wave
    decaying_sine_wave
    chirp
    chirp_centered
    gaussian_pulse
    picket_fence
    heavi_sine
    bumps


.. currentmodule:: cr.nimble.dsp



Discrete Cosine Transform
-------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    dct
    idct
    orthonormal_dct
    orthonormal_idct

.. currentmodule:: cr.nimble.dsp

Fast Walsh Hadamard Transform
------------------------------

There is no separate Inverse Fast Walsh Hadamard Transform as FWHT is the inverse of
itself except for a normalization factor.
In other words,  ``x == fwht(fwht(x)) / n`` where n is the length of x.

.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    fwht

Utilities
-----------------------


.. autosummary::
    :nosignatures:
    :toctree: _autosummary

    time_values

