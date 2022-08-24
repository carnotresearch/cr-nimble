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

    scale_to_0_1s
    normalize

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


Artificial Noise
-----------------------------------


.. autosummary::
  :toctree: _autosummary

  awgn_at_snr


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

