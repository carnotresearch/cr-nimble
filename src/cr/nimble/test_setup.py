import pytest

import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)


from jax.config import config
config.update("jax_enable_x64", True)


import jax
import jax.numpy as jnp
from jax import random, lax, vmap, jit

import cr.nimble as cnb

rtol = 1e-8 if jax.config.jax_enable_x64 else 1e-6
atol = 1e-7 if jax.config.jax_enable_x64 else 1e-5
decimal_cmp = 7 if jax.config.jax_enable_x64 else 6

float_type = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
complex_type = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
