from ls_setup import *

def test_jacobi():
    A = jnp.array([[3., 2], [2, 6]])
    b = jnp.array([2., -8])
    sol = spd.jacobi_solve_jit(A, b)
    r = A @ sol.x - b
    r_norm_sqr = r.T @ r
    assert jnp.isclose(0, r_norm_sqr, atol=1e-4)
