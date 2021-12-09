from cr.nimble.test_setup import *

M = 10
p = 3
N = 5

A = jnp.zeros([M, p])
B = jnp.ones([M, p])

C = A.T
D = B.T

def test_1():
    cnb.pairwise_sqr_l2_distances_rw(A, B)

def test_2():
    cnb.pairwise_sqr_l2_distances_cw(A, B)

def test_a():
    cnb.pairwise_l2_distances_rw(A, B)

def test_b():
    cnb.pairwise_l2_distances_cw(A, B)

def test_c():
    cnb.pdist_sqr_l2_rw(A)

def test_d():
    cnb.pdist_sqr_l2_cw(A)

def test_e():
    cnb.pdist_l2_rw(A)

def test_f():
    cnb.pdist_l2_cw(A)


def test_3():
    cnb.pairwise_l1_distances_rw(C, D)

def test_4():
    cnb.pairwise_l1_distances_cw(A, B)

def test_5():
    cnb.pdist_l1_rw(C)

def test_6():
    cnb.pdist_l1_cw(A)

def test_7():
    cnb.pairwise_linf_distances_rw(C, D)


def test_8():
    cnb.pairwise_linf_distances_cw(A, B)

def test_9():
    cnb.pdist_linf_rw(C)

def test_10():
    cnb.pdist_linf_cw(B)
