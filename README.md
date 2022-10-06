# CR-Nimble

`CR-Nimble` consists of fast linear algebra
and signal processing routines.
Most of the routines have been implemented using
Google JAX. Thus, they can be easily run on
a variety of hardware (CPU, GPU, TPU).

Functionality includes:

* Utility functions for working with vectors, matrices and arrays
* Linear algebra functions
* Digital signal processing functions
* Data compression functions
* Test data generation functions


Installation

```{shell}
python -m pip install cr-nimble
```

For Windows, you can use unofficial JAX builds
from [here](https://github.com/cloudhan/jax-windows-builder).

Import

```{python}
import cr.nimble as crn
```

See [documentation](https://cr-nimble.readthedocs.io)
for library usage.

`CR-Nimble` is part of
[CR-Suite](https://carnotresearch.github.io/cr-suite/).

Related libraries:

* [CR-Wavelets](https://cr-wavelets.readthedocs.io)
* [CR-Sparse](https://cr-sparse.readthedocs.io)


[![codecov](https://codecov.io/gh/carnotresearch/cr-nimble/branch/main/graph/badge.svg?token=PX1MGTZ7VL)](https://codecov.io/gh/carnotresearch/cr-nimble) 
[![Unit Tests](https://github.com/carnotresearch/cr-nimble/actions/workflows/ci.yml/badge.svg)](https://github.com/carnotresearch/cr-nimble/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/cr-nimble/badge/?version=latest)](https://cr-nimble.readthedocs.io/en/latest/?badge=latest)
