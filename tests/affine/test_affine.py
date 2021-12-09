from cr.nimble.test_setup import *

import cr.nimble.affine

homogenize = jax.jit(cnb.affine.homogenize)
homogenize_vec = jax.jit(cnb.affine.homogenize_vec)
homogenize_cols = jax.jit(cnb.affine.homogenize_cols)

def test_homogenize_vec():
    x = jnp.array([1,2,3])
    y = homogenize_vec(x)
    assert len(x) + 1 == len(y)
    assert y[-1] == 1

def test_homogenize_vec2():
    x = jnp.array([1,2,3])
    y = homogenize(x)
    assert len(x) + 1 == len(y)
    assert y[-1] == 1

def test_homogenize_cols():
    x = jnp.array([[1,2,3],[4,5,6]])
    y = homogenize_cols(x)
    assert x.shape[0] + 1 == y.shape[0]
    assert jnp.allclose(y[-1, :] , 1)


def test_homogenize_cols2():
    x = jnp.array([[1,2,3],[4,5,6]])
    y = homogenize(x)
    assert x.shape[0] + 1 == y.shape[0]
    assert jnp.allclose(y[-1, :] , 1)
