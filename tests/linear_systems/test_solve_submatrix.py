from ls_setup import *

@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_solve1(K):
    M = 20
    N = 40
    Phi = cnb.gaussian_mtx(cnb.KEYS[0], M, N)
    cols = random.permutation(cnb.KEYS[1], jnp.arange(N))[:K]
    X = random.normal(cnb.KEYS[2], (K, 1))
    Phi_I = Phi[:, cols]
    B_ref = Phi_I @ X
    B = cnb.mult_with_submatrix(Phi, cols, X)
    assert_allclose(B_ref, B)
    Z, R = cnb.solve_on_submatrix(Phi, cols, B)
    assert_allclose(Z, X, atol=atol, rtol=rtol)


submat_multiplier = vmap(cnb.mult_with_submatrix, (None, 1, 1), 1)
submat_solver = vmap(cnb.solve_on_submatrix, (None, 1, 1), (1, 1,))

@pytest.mark.parametrize("K", [1, 2, 4])
def test_solve2(K):
    M = 20
    N = 40
    Phi = cnb.gaussian_mtx(cnb.KEYS[0], M, N)
    # Number of signals
    S = 4
    # index sets for each signal
    omega = jnp.arange(N)
    keys = random.split(cnb.KEYS[1], S)
    set_gen = lambda key : random.permutation(key, omega)[:K] 
    cols = vmap(set_gen, 0, 1)(keys)
    # signals [column wise]
    X = random.normal(cnb.KEYS[2], (K, S))
    # measurements
    B = submat_multiplier(Phi, cols, X)
    # solutions
    Z, R = submat_solver(Phi, cols, B)
    # verify
    assert_allclose(Z, X, atol=atol, rtol=rtol)
