import numpy as np
import matplotlib.pyplot as plt

from jdmat import dmat_build_tprod_natural_pruned
from jquad_tprod import jquad_mapped_build_kappa
from jbasis import (
  jbasis_build_structures,
  jbasis_eval_all,
  jbasis_eval_all_with_grad,
)


def spy_mat(A, tol, title):
  S = np.abs(A) > tol
  plt.figure(figsize=(6, 6))
  #plt.spy(S, markersize=1)
  plt.spy(A)
  plt.title(title)
  plt.xlabel("source index j")
  plt.ylabel("range index i")
  plt.tight_layout()
  plt.show()


def normalize_w(w):
  sw = float(np.sum(w))
  if sw != 0.0:
    return w / sw
  return w


def rel_coeff_err(c1, c2):
  return np.linalg.norm(c1 - c2) / (np.linalg.norm(c2) + 1e-300)


def rel_l2_err_from_coeffs(V, w, c1, c2):
  u1 = V @ c1
  u2 = V @ c2
  d = u1 - u2
  num = np.sum(w * d * d)
  den = np.sum(w * u2 * u2) + 1e-300
  return float(np.sqrt(num / den))


def dk_natural(D, axis):
  dk = np.zeros(D + 1, dtype=np.float64)
  dk[axis] = 1.0
  dk[D] = 1.0
  return dk


def run_one(D, n, q, seed=123, speye_tol=1e-10):
  rng = np.random.default_rng(seed)
  kappa_src = np.full(D + 1, 0.5, dtype=np.float64)

  alpha_src, tail_src, invh_src = jbasis_build_structures(D, n, kappa_src)
  M = alpha_src.shape[0]
  c_src = rng.standard_normal(M).astype(np.float64)

  print(f"\n===== D={D} ===== n={n} q={q} M={M}")

  for axis in range(D):
    dk = dk_natural(D, axis)
    kappa_rng = kappa_src + dk

    # Build pruned operator from C++
    Dp = dmat_build_tprod_natural_pruned(D, n, q, kappa_src, axis)

    # Quadrature & basis eval in range space (must match κ_rng)
    X, w = jquad_mapped_build_kappa(D, q, kappa_rng)

    alpha_rng, tail_rng, invh_rng = jbasis_build_structures(D, n, kappa_rng)
    Vrng = jbasis_eval_all(X, kappa_rng, n, alpha_rng, tail_rng, invh_rng, D)

    # Source values + grads
    Vsrc, dVsrc = jbasis_eval_all_with_grad(X, kappa_src, n, alpha_src, tail_src, invh_src, D)

    # Pointwise derivative of u_n
    du_pts = dVsrc[:, :, axis] @ c_src

    # Project derivative samples into κ_rng basis
    c_proj = Vrng.T @ (w * du_pts)

    # Apply operator
    c_from = Dp @ c_src

    err_c = rel_coeff_err(c_from, c_proj)
    err_l2 = rel_l2_err_from_coeffs(Vrng, w, c_from, c_proj)

    axname = "xyzt"[axis] if axis < 4 else f"x{axis}"
    print(f"  d/d{axname}: dk={dk.tolist()}  rel_coeff={err_c:.3e}  rel_L2={err_l2:.3e}")
    spy_mat(
      Dp,
      speye_tol,
      f"D={D} d/d{axname} pruned C++  |D|>{speye_tol:g}\n"
      f"dk={dk.tolist()}  rel_coeff={err_c:.2e}  rel_L2={err_l2:.2e}"
    )


def main():
  cfg = {
    1: (18, 24),
    2: (10, 16),
    3: (7,  13),
    4: (5,  11),
  }
  for D in (1, 2, 3, 4):
    n, q = cfg[D]
    run_one(D, n, q, seed=123, speye_tol=1e-14)


if __name__ == "__main__":
  main()

