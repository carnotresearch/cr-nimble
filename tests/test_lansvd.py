from nb_setup import *


def test_lanbpro1():
    A =  jnp.eye(4)
    r = nbsvd.lanbpro_random_start(cnb.KEYS[0], A)
    state = nbsvd.lanbpro_jit(A, 4, r)
    s = str(state)
    assert_allclose(state.alpha, 1., atol=atol)
    assert_allclose(state.beta[1:], 0., atol=atol)


def test_lansvd1():
    A =  jnp.eye(4)
    r = nbsvd.lanbpro_random_start(cnb.KEYS[0], A)
    U, S, V, bnd, n_converged, state = nbsvd.lansvd_simple_jit(A, 4, r)
    assert_allclose(state.alpha, 1., atol=atol)
    assert_allclose(state.beta[1:], 0., atol=atol)
