import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from jkmat import kmat_build_tprod, kmat_build_tprod_pruned_csc
from jbasis import jbasis_build_structures, jbasis_eval_all 
from jquad_tprod import jquad_mapped_build_kappa            
from libjpolyd_loader import libjpolyd


def add_degree_block_boxes(D, n, ax=None, linewidth=1.0):
  """
  Overlay boxes for each total-degree block Hom(j) on a Pi_n-ordered matrix plot.

  Assumes global ordering is graded by total degree, so:
    block_start(j) = dim_Pi(D, j-1)
    block_end(j)   = dim_Pi(D, j)
    block_size(j)  = dim_Pi(D, j) - dim_Pi(D, j-1)

  Uses: libjpolyd.jbasis_dim_Pi(D, n)
  """
  if ax is None:
    ax = plt.gca()

  dim_Pi = libjpolyd.jbasis_dim_Pi

  # Draw diagonal block boxes and boundary lines
  for j in range(n + 1):
    b0 = 0 if j == 0 else int(dim_Pi(int(D), int(j - 1)))
    b1 = int(dim_Pi(int(D), int(j)))
    size = b1 - b0

    # diagonal block box (b0:b1, b0:b1)
    rect = patches.Rectangle(
      (b0 - 0.5, b0 - 0.5),
      size,
      size,
      fill=False,
      linewidth=linewidth,
      edgecolor="k",
    )
    ax.add_patch(rect)

  # Degree boundaries (vertical/horizontal lines)
  for j in range(n):
    b = int(dim_Pi(int(D), int(j)))
    ax.axvline(b - 0.5, linewidth=linewidth * 0.6, color="k")
    ax.axhline(b - 0.5, linewidth=linewidth * 0.6, color="k")

def _l2_err_weighted(u, v, w):
  # w already includes the κ-weight (build_kappa), so this is weighted L2.
  d = u - v
  num = np.sum(w * d * d)
  den = np.sum(w * v * v) + 1e-300
  return np.sqrt(num / den)


def test_kmat(D, n, a=0.0, tol_spy=1e-12, seed=123, q=None):
  # Promote kappa -> kappa+1 (all components)
  #kappa_src = np.full(D + 1, a, dtype=np.float64)
  #dkappa = np.full(D+1, 0, dtype=np.float64)
  kappa_src = -0.25 + (5.0 + 0.25) * np.random.rand(D + 1)
  kappa_tgt = kappa_src.copy()
  kappa_tgt[-1] = kappa_tgt[-1]+1 

  # Quadrature points per axis: safe default q=n+2 for extra margin
  if q is None:
    q = n + 2

  # Build K using internal κ-aware mapped quadrature
  K = kmat_build_tprod_pruned_csc(D, n, q, kappa_src, kappa_tgt)
  #K = kmat_build_tprod(D, n, q, kappa_src, kappa_tgt)
  
  # --- Sparsity plot ---
  S = K#np.abs(K) > tol_spy
  plt.figure(figsize=(6, 6))
  #plt.spy(S.toarray(), markersize=1)
  plt.spy(S, markersize=1)
  ax = plt.gca()
  add_degree_block_boxes(D, n, ax=ax, linewidth=1.0)

  plt.title(f"K sparsity (|K|>{tol_spy:g})  D={D}, n={n}, a→a + (1,0..), q={q}")
  plt.xlabel("source index")
  plt.ylabel("target index")
  plt.tight_layout()
  plt.show()

  # --- L2 correctness check in L2(w_tgt) ---
  # Evaluate both expansions at κ_tgt-weighted quadrature nodes
  X, w = jquad_mapped_build_kappa(D, q, kappa_tgt)  # weights include κ_tgt weight 

  # Basis eval for src and tgt spaces at the same points
  alpha_s, tail_s, invh_s = jbasis_build_structures(D, n, kappa_src)
  alpha_t, tail_t, invh_t = jbasis_build_structures(D, n, kappa_tgt)

  Vsrc = jbasis_eval_all(X, kappa_src, n, alpha_s, tail_s, invh_s, D)  # shape (npts,M), Fortran order 
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
  #H_diag = np.sum(w[:, None] * (Vtgt * Vtgt), axis=0)
  #print(f"[D={D}] ortho check: min(diag(H))={H_diag.min():.6f}, max(diag(H))={H_diag.max():.6f}")

  return rel


if __name__ == "__main__":
  # D=2 test
  n = 3
  q = n + 1
  test_kmat(D=1, n=n, a=0.5, tol_spy=1e-14, seed=1, q=q)
  test_kmat(D=2, n=n, a=0.5, tol_spy=1e-14, seed=1, q=q)
  # D=3 test (keep n smaller to avoid big dense K)
  test_kmat(D=3, n=n, a=0.5, tol_spy=1e-14, seed=3, q=q)
  
  # D=4 test (keep n smaller to avoid big dense K)
  test_kmat(D=4, n=n, a=0.5, tol_spy=1e-14, seed=2, q=q)

