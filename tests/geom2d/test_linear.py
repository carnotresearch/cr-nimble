from cr.nimble.test_setup import *

def test_point():
    x = cnb.point2d(1, 2)

def test_vec():
    x = cnb.vec2d(1, 2)

def test_rotate2d_cw():
    theta = jnp.pi
    R = cnb.rotate2d_cw(theta)

def test_rotate2d_ccw():
    theta = jnp.pi
    R = cnb.rotate2d_ccw(theta)

def test_reflect2d():
    theta = jnp.pi
    R = cnb.reflect2d(theta)
