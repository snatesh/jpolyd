import numpy as np
from jquad_tprod import *
from jweight import *
from jbasis import *

def gram_error(D, n, nquad, kappa):
    alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)
    pts, w = jquad_mapped_build_kappa(D, nquad, kappa)
    V = jbasis_eval_all(pts, kappa, n, alpha_table, tail_deg, inv_h, D)  # (Npts x M)
    G = V.T @ (V * w[:, None])
    E = G - np.eye(G.shape[0])
    return np.linalg.norm(E, 'fro'), np.max(np.abs(E))

def main():
    rng = np.random.default_rng(7)
    ok = True
    cases = []

    # Deterministic kappas per D (well away from -1/2)
    fixed = {
        2: [np.array([1.7, 3.3, 2.8])],
        3: [np.array([2.5, 1.5, 5.0, 0.9])],
        4: [np.array([2.5, 1.5, 5.5, 3.5, 6.5])],
        5: [np.array([2.5, 1.5, 5.5, 3.5, 6.5, 1.7])],
    }

    for D in range(2, 6):           # D = 2..5
        n = 5
        nquad = n + 1               # exactness ~ 2(n+1)-1
        # a couple random kappas (>-0.49 for safety), plus fixed
        K = [0.6 + 6*rng.random(D+1) for _ in range(2)]
        K = [np.asarray(k, float) for k in K]
        K.extend(fixed.get(D, []))

        for kappa in K:
            frob, mx = gram_error(D, n, nquad, kappa)
            cases.append((D, n, nquad, kappa, frob, mx))
            # Keep a modest tolerance; with nquad=n+1 this should be very small
            tol = 1e-11 if D <= 3 else 5e-11
            ok &= (frob < 1e-10) and (mx < tol)

    # Report
    for D, n, nquad, kappa, frob, mx in cases:
        print(f"D={D} n={n} nquad={nquad}  kappa={kappa}  "
              f"||G-I||_F={frob:.3e}  max|G-I|={mx:.3e}")
    print("RESULT:", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    main()
