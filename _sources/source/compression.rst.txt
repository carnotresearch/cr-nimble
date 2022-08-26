.. _api:compression:

Data Compression
===============================

.. contents::
    :depth: 2
    :local:


The library comes with some basic data compression
routines. They are helpful in simple use cases like:

- compression of binary arrays
- run length encoding
- fixed length encoding of integers

These routines are primarily based on
`numpy` arrays and `bitarray` based
compressed bit arrays. This module
doesn't use JAX.


.. currentmodule:: cr.nimble.compression


.. autosummary::
    :toctree: _autosummary

    count_runs_values
    expand_runs_values
    encode_int_arr_sgn_mag_fl
    decode_int_arr_sgn_mag_fl
    count_binary_runs
    encode_binary_arr
    decode_binary_arr
    binary_compression_ratio
    binary_space_saving_ratio

