import numpy as np
from jmat import *
from jbasis import *
from jquad_tprod import *


def comb(n, k):
    # simple integer n choose k
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= n - (k - i)
        den *= i
    return num // den


def dim_R(d, n):
    # dim of homogeneous polynomials of degree n in d dims
    # r_n^d = C(n + d - 1, n)
    return comb(n + d - 1, n)


def test_jacobi_operator(D=3, n=5, kappa=None, nquad_test=None):
    if kappa is None:
        # unit-ish weight default
        kappa = np.array([0.5] * (D + 1), dtype=np.float64)

    if nquad_test is None:
        nquad_test = n + 1   # slightly higher for testing

    # build structures (alpha_table, tail_deg, inv_h)
    alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)

    # total number of basis functions
    M = alpha_table.shape[0]

    # index range for homogeneous degree n block
    # since build_alpha_table loops total degree 0..n in order,
    # block offsets are just cumulative dim_R
    offset_deg = [0]
    for m in range(1, n + 1):
        offset_deg.append(offset_deg[-1] + dim_R(D, m - 1))
    # last block (degree n) starts at:
    start_n = offset_deg[-1]
    Rn = dim_R(D, n)  # size of last block

    # Build Jacobi matrices J_i (D, M, M)
    J_all = jmat_build(D, n, kappa)

    # Build test quadrature for the kappa-weighted measure
    X_test, W_test = jquad_mapped_build_kappa(D, nquad_test, kappa)
    npts = X_test.shape[0]

    # Evaluate basis at quadrature points
    V = jbasis_eval_all(X_test, kappa, n, alpha_table, tail_deg, inv_h, D)
    # V: (npts, M)

    # Choose random coefficients supported only in degrees <= n-1
    rng = np.random.default_rng(12345)
    c = rng.standard_normal(M)
    c[start_n : start_n + Rn] = 0.0  # zero out highest-degree block

    # precompute u(x) once
    u = V @ c   # shape (npts,)

    # We'll measure L2_kappa norms via the test quadrature
    W_col = W_test[:, None]  # npts x 1

    residuals = []
    for i in range(D):
        # coordinate x_i (0-based indexing)
        xi = X_test[:, i]

        # left side: x_i * u
        xu = xi * u

        # right side: projection via J_i
        Ji_c = J_all[i] @ c       # length M
        u_proj = V @ Ji_c         # length npts

        r = xu - u_proj
        # approximate L2_kappa norm via quadrature
        # ||r||^2 ≈ sum w * r^2
        r2 = (r * r) * W_test
        err_L2 = np.sqrt(r2.sum())

        residuals.append(err_L2)

    print(f"D={D}, n={n}, kappa={kappa}")
    print(f"nquad_test={nquad_test}")
    for i, e in enumerate(residuals):
        print(f"  L2_kappa error for x_{i+1} u - J_{i+1} u : {e:.3e}")

    return residuals


if __name__ == "__main__":
    # a few quick runs
    print("=== unit-ish weight, D=3, n=5 ===")
    test_jacobi_operator(D=3, n=5, kappa=np.array([0.5, 0.5, 0.5, 0.5]))

    print("\n=== nontrivial kappa, D=3, n=5 ===")
    test_jacobi_operator(D=3, n=5,
                         kappa=np.array([1.7, 3.3, 2.8, 0.9]))
    print("\n=== higher D annd n, D=4, n=8 ===")
    test_jacobi_operator(D=4, n=8,
                         kappa=np.array([1.7, 3.3, 2.8, 0.9, 2.2]))


