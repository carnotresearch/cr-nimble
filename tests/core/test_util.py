from core_setup import *


def test_promote_arg_dtypes():
    res = cnb.promote_arg_dtypes(jnp.array(1), jnp.array(2))
    expected = jnp.array([1.0, 2.0])
    assert jnp.array_equal(res, expected)
    assert jnp.array_equal(cnb.promote_arg_dtypes(jnp.array(1)), jnp.array(1.))
    cnb.promote_arg_dtypes(jnp.array(1), jnp.array(2.))

def test_canonicalize_dtype():
    cnb.canonicalize_dtype(None)
    cnb.canonicalize_dtype(jnp.int32)


def test_is_cpu():
    assert_equal(cnb.is_cpu(), cnb.platform == 'cpu')

def test_is_gpu():
    assert_equal(cnb.is_gpu(), cnb.platform == 'gpu')

def test_is_tpu():
    assert_equal(cnb.is_tpu(), cnb.platform == 'tpu')

def test_check_shapes_are_equal():
    z = jnp.zeros(4)
    o = jnp.ones(4)
    cnb.check_shapes_are_equal(z, o)
    o = jnp.ones(5)
    with assert_raises(ValueError):
        cnb.check_shapes_are_equal(z, o)

def test_promote_to_complex():
    z = jnp.zeros(4)
    z = cnb.promote_to_complex(z)
    assert z.dtype == np.complex64

def test_promote_to_real():
    z = jnp.zeros(4, dtype=int)
    z = cnb.promote_to_real(z)
    assert z.dtype == np.float32


def test_nbytes_live_buffers():
    nbytes = cnb.nbytes_live_buffers()
    assert nbytes > 0