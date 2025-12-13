import numpy as np
import matplotlib.pyplot as plt

from jkmat import kmat_build_tprod
from jbasis import jbasis_build_structures, jbasis_eval_all  # :contentReference[oaicite:2]{index=2}
from jquad_tprod import jquad_mapped_build_kappa                # :contentReference[oaicite:3]{index=3}


def _l2_err_weighted(u, v, w):
  # w already includes the κ-weight (build_kappa), so this is weighted L2.
  d = u - v
  num = np.sum(w * d * d)
  den = np.sum(w * v * v) + 1e-300
  return np.sqrt(num / den)


def test_kmat(D, n, a=0.0, tol_spy=1e-12, seed=123, q=None):
  # Promote kappa -> kappa+1 (all components)
  kappa_src = np.full(D + 1, a, dtype=np.float64)
  kappa_tgt = np.full(D + 1, a + 1.0, dtype=np.float64)

  # Quadrature points per axis: safe default q=n+2 for extra margin
  if q is None:
    q = n + 2

  # Build K using internal κ-aware mapped quadrature
  K = kmat_build_tprod(D, n, q, kappa_src, kappa_tgt)

  # --- Sparsity plot ---
  S = K#np.abs(K) > tol_spy
  plt.figure(figsize=(6, 6))
  plt.spy(S, markersize=1)
  plt.title(f"K sparsity (|K|>{tol_spy:g})  D={D}, n={n}, a→a+1, q={q}")
  plt.xlabel("source index")
  plt.ylabel("target index")
  plt.tight_layout()
  plt.show()

  # --- L2 correctness check in L2(w_tgt) ---
  # Evaluate both expansions at κ_tgt-weighted quadrature nodes
  X, w = jquad_mapped_build_kappa(D, q, kappa_tgt)  # weights include κ_tgt weight :contentReference[oaicite:4]{index=4}

  # Basis eval for src and tgt spaces at the same points
  alpha_s, tail_s, invh_s = jbasis_build_structures(D, n, kappa_src)
  alpha_t, tail_t, invh_t = jbasis_build_structures(D, n, kappa_tgt)

  Vsrc = jbasis_eval_all(X, kappa_src, n, alpha_s, tail_s, invh_s, D)  # shape (npts,M), Fortran order :contentReference[oaicite:5]{index=5}
  Vtgt = jbasis_eval_all(X, kappa_tgt, n, alpha_t, tail_t, invh_t, D)

  M = Vsrc.shape[1]
  assert Vtgt.shape[1] == M

  rng = np.random.default_rng(seed)
  c_src = rng.standard_normal(M)

  c_tgt = K @ c_src

  # Evaluate u(x) in both representations
  u_src = Vsrc @ c_src
  u_tgt = Vtgt @ c_tgt

  rel = _l2_err_weighted(u_tgt, u_src, w)

  print(f"[D={D}] rel L2(w_tgt) error = {rel:.3e}")

  # Optional: also check that tgt basis is orthonormal under (X,w)
  # H ~ I
  # (this is a good sanity check that the quadrature + inv_h are consistent)
  H_diag = np.sum(w[:, None] * (Vtgt * Vtgt), axis=0)
  print(f"[D={D}] ortho check: min(diag(H))={H_diag.min():.6f}, max(diag(H))={H_diag.max():.6f}")

  return rel


if __name__ == "__main__":
  # D=2 test
  n = 12
  q = n + 2
  test_kmat(D=2, n=n, a=0.5, tol_spy=1e-14, seed=1, q=q)

  # D=3 test (keep n smaller to avoid big dense K)
  test_kmat(D=3, n=n, a=0.5, tol_spy=1e-14, seed=3, q=q)

