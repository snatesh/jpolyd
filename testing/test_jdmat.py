# test_jdmat.py
#
# Tests the “natural” parameter shift rule for differentiation in D=1..4:
#   kappa_src = (1/2,...,1/2) in R^{D+1}
#   for axis i=0..D-1:
#     dk = e_i + e_{D}   (i.e., increment axis parameter and the last/barycentric parameter)
#     kappa_rng = kappa_src + dk
#
# Correctness check (no “true derivative” of u needed):
#   1) pick random modal coeffs c_src in basis(kappa_src)
#   2) evaluate du/dx_axis at quadrature points using analytic gradient branch of eval_all
#   3) project that pointwise derivative into basis(kappa_rng) -> c_proj
#   4) compare c_proj against (Dmat @ c_src), where Dmat is built by js_dmat_build_tprod
#
# Also plots sparsity (spy) for each Dmat.

import numpy as np
import matplotlib.pyplot as plt

from jdmat import dmat_build_tprod
from jquad_tprod import jquad_mapped_build_kappa

# NOTE: we use the gradient-capable evaluator
from jbasis import (
  jbasis_build_structures,
  jbasis_eval_all,
  jbasis_eval_all_with_grad,
)


def spy_mat(A, tol, title):
  S = np.abs(A) > tol
  plt.figure(figsize=(6, 6))
  plt.spy(S, markersize=1)
  plt.title(title)
  plt.xlabel("source index j")
  plt.ylabel("range index i")
  plt.tight_layout()
  plt.show()

def prune_matrix(Dmat, tol):
  Dp = Dmat.copy()
  Dp[np.abs(Dp) < tol] = 0.0
  return Dp

def rel_coeff_err(c1, c2):
  return np.linalg.norm(c1 - c2) / (np.linalg.norm(c2) + 1e-300)


def rel_l2_err_from_coeffs(V, w, c1, c2):
  u1 = V @ c1
  u2 = V @ c2
  d = u1 - u2
  num = np.sum(w * d * d)
  den = np.sum(w * u2 * u2) + 1e-300
  return float(np.sqrt(num / den))


def normalize_w(w):
  sw = float(np.sum(w))
  if sw != 0.0:
    return w / sw
  return w


def dk_natural(D, axis):
  dk = np.zeros(D + 1, dtype=np.float64)
  dk[axis] = 1.0
  dk[D] = 1.0
  return dk


def run_one(D, n, q, seed=0, tol_spy=1e-14):
  rng = np.random.default_rng(seed)

  kappa_src = np.full(D + 1, 0.5, dtype=np.float64)

  # Build source basis structures and random coefficients
  alpha_src, tail_src, invh_src = jbasis_build_structures(D, n, kappa_src)
  M = alpha_src.shape[0]
  c_src = rng.standard_normal(M).astype(np.float64)

  print(f"\n===== D={D} =====")
  print(f"n={n}, q={q}, M={M}")

  for axis in range(D):
    dk = dk_natural(D, axis)
    kappa_rng = kappa_src + dk

    # Build Dmat with hard-coded natural shift
    Dmat = dmat_build_tprod(D, n, q, kappa_src, kappa_rng, axis)

    # Quadrature for range weight (must match how dmat is built)
    X, w = jquad_mapped_build_kappa(D, q, kappa_rng)
    w = normalize_w(w)

    # Range basis values at X
    alpha_rng, tail_rng, invh_rng = jbasis_build_structures(D, n, kappa_rng)
    Vrng = jbasis_eval_all(X, kappa_rng, n, alpha_rng, tail_rng, invh_rng, D)

    # Source basis values + analytic gradients at X
    Vsrc, dVsrc = jbasis_eval_all_with_grad(X, kappa_src, n, alpha_src, tail_src, invh_src, D)
    # dVsrc shape: (npts, M, D)

    # Pointwise derivative of u_n(x) = sum_j c_src[j] phi_j(x)
    du_pts = dVsrc[:, :, axis] @ c_src  # (npts,)

    # Project pointwise derivative into range basis:
    # c_proj[i] = sum_p w[p] * du[p] * phi_rng_i(x_p)
    c_proj = Vrng.T @ (w * du_pts)

    # Coeffs from operator
    c_from = Dmat @ c_src


    # Errors: coeff space + L2(w_rng)
    err_c = rel_coeff_err(c_from, c_proj)
    err_l2 = rel_l2_err_from_coeffs(Vrng, w, c_from, c_proj)
    
    axname = "xyzt"[axis] if axis < 4 else f"x{axis}"
    print(f"  d/d{axname}: dk={dk.tolist()}  rel_coeff={err_c:.3e}  rel_L2={err_l2:.3e}")
    
    # ---- Pruned operator test ----
    tol_prune = 1e-10
    Dmat_p = prune_matrix(Dmat, tol_prune)

    c_from_p = Dmat_p @ c_src

    err_c_pruned = rel_coeff_err(c_from_p, c_proj)
    err_l2_pruned = rel_l2_err_from_coeffs(Vrng, w, c_from_p, c_proj)

    print(
      f"    pruned(tol={tol_prune:g}): "
      f"rel_coeff={err_c_pruned:.3e}  rel_L2={err_l2_pruned:.3e}"
    )

    spy_mat(
      Dmat,
      tol_spy,
      f"D={D} d/d{axname}  dk={dk.tolist()}  |D|>{tol_spy:g}\n"
      f"n={n}, q={q}, rel_coeff={err_c:.2e}, rel_L2={err_l2:.2e}"
    )


def main():
  # Keep these modest; increase once everything is stable.
  # q should be a bit larger than n to make the projection-based operators sharp.
  cfg = {
    1: (18, 24),
    2: (10, 16),
    3: (10,  12),
    4: (5,  11),
  }

  for D in (1, 2, 3, 4):
    n, q = cfg[D]
    run_one(D, n, q, seed=123, tol_spy=1e-10)


if __name__ == "__main__":
  main()

