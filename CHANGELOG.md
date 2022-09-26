# Change Log
All notable changes to this project will be documented in this file.

* This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
* The format of this log is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [Unreleased]

[Documentation](https://cr-nimble.readthedocs.io/en/latest/)


## [0.3.1] - 2022-09-10

[Documentation](https://cr-nimble.readthedocs.io/en/v0.3.1/)

### Added

Metrics

- normalized_mse
- percent_rms_diff
- compression_ratio
- cr_to_pss
- pss_to_cr

Noise

- awgn_at_snr_std

Matrices

- mat_column_blocks
- block_diag

Vectors

- has_equal_values_vec

Special matrices

- toeplitz_mat
- toeplitz_mult
- circulant_mat
- circulant_mult


Misc

- to_tex_matrix



## [0.3.0] - 2022-08-27

[Documentation](https://cr-nimble.readthedocs.io/en/v0.3.0/)

### Added

Data Compression

- Binary data encoding/decoding
- Run length encoding/decoding
- Fixed length encoding/decoding

Digital Signal Processing

- Scaling functions
- Quantized
- Energy fraction based thresholding

Metrics

- Percentage root mean square difference

### Removed

### Changed

- Statistical normalization renamed with changes in return type
- Digital signal processing related functions moved under
  `cr.nimble.dsp`


### Improved

- Documentation improved
- API organization improved


## [0.2.4] - 2022-08-17

[Documentation](https://cr-nimble.readthedocs.io/en/v0.2.4/)


### Added

- Digital signal processing utilities moved from cr-sparse to cr-nimble
- Moved discrete number related functions from cr-sparse.
- Some sparse vector and matrix processing functionality moved from cr-sparse.

### Removed

- Unnecessary `__init__.py` files removed.

### Notes

- Jax 0.3.14 compatibility
- Aligning version numbering across sister projects.

## [0.1.1] - 2021-12-07

### Added

- distance, matrix, ndarray, metrics, modules were moved from `cr-sparse` to `cr-nimble`
- some more vector functions were moved from `cr-sparse` to `cr-nimble`

### Improved

- All unit tests were moved to 64-bit floating point data.


## [0.1.0] - 2021-12-07

Initial release by refactoring code from `cr-nimble`.


[Unreleased]: https://github.com/carnotresearch/cr-nimble/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/carnotresearch/cr-nimble/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/carnotresearch/cr-nimble/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/carnotresearch/cr-nimble/compare/v0.1.1...v0.2.4
[0.1.1]: https://github.com/carnotresearch/cr-nimble/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/carnotresearch/cr-nimble/releases/tag/v0.1.0