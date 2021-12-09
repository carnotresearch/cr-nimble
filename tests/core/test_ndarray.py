from core_setup import *


def test_arr_largest_index():
    x = jnp.array([1, -2, -3, 2])
    assert_equal(cnb.arr_largest_index(x), (jnp.array([2]), ))
    x = jnp.reshape(x, (2,2))
    idx = cnb.arr_largest_index(x)
    print(idx)
    assert_equal(idx, (jnp.array([1]), jnp.array([0])))


def test_arr_l2norm():
    x = jnp.zeros(10)
    assert_allclose(cnb.arr_l2norm(x), 0.)

def test_arr_l2norm_sqr():
    x = jnp.zeros(10)
    assert_allclose(cnb.arr_l2norm_sqr(x), 0.)

def test_arr_vdot():
    x = jnp.zeros(10)
    assert_allclose(cnb.arr_vdot(x, x), 0.)




@pytest.mark.parametrize("x,y", [ ([1, 0], [1, 0]),
([1], [1, 0]),
([1 + 0j, 0], [1+0j, 0]),
([1 + 1j, 0], [1+1j, 0]),
([1 + 1j, 2-3j], [1, 0]),
([1, -4], [1+2j, -3j]),
])
def test_rdot(x, y):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x1 = cnb.arr2vec(x)
    y1 = cnb.arr2vec(y)
    expected = jnp.sum(jnp.conjugate(x1) * y1)
    expected = jnp.real(expected)
    assert_almost_equal(cnb.arr_rdot(x, y), expected)

