from dsp_setup import *

atol = 1e-6
rtol = 1e-6


z = jnp.zeros((4,4))
tau = 0.5
perc = 20.
o = jnp.ones((4,4))

def test_hard_threshold():
    assert_allclose(z, dsp.hard_threshold_tau(z, tau))


def test_soft_threshold():
    assert_allclose(z, dsp.soft_threshold_tau(z, tau))
    assert_allclose(z, dsp.soft_threshold_tau(z+0j, tau))

def test_half_threshold():
    assert_allclose(z, dsp.half_threshold_tau(z, tau))



def test_hard_threshold_percentile():
    assert_allclose(z, dsp.hard_threshold_percentile(z, perc))


def test_soft_threshold_percentile():
    assert_allclose(z, dsp.soft_threshold_percentile(z, perc))
    assert_allclose(z, dsp.soft_threshold_percentile(z+0j, perc))

def test_half_threshold_percentile():
    f = 2/3 * o
    assert_allclose(f, dsp.half_threshold_percentile(o, perc))


def test_gamma_to_tau():
    gamma = 1.
    tau = dsp.gamma_to_tau_half_threshold(gamma)
    tau = dsp.gamma_to_tau_hard_threshold(gamma)