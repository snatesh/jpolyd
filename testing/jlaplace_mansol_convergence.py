#!/usr/bin/env python3
"""
Convergence plot for the jlaplace manufactured-solution test.

Run from the repo root after installing/building the Python wrappers, e.g.

  python testing/jlaplace_mansol_convergence.py --D 3 --n-min 2 --n-max 9

For D=4, keep the sweep modest:

  python testing/jlaplace_mansol_convergence.py --D 4 --n-min 2 --n-max 7 --q-pad 2

The script reuses helper functions from testing/jlaplace_lint_test.py, so keep
this file in the same testing/ directory as jlaplace_lint_test.py.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
  sys.path.insert(0, _THIS_DIR)

from jlaplace_lint_test import (  # noqa: E402
  affine_from_verts,
  build_basis_structs,
  build_reference_second_partials,
  laplacian_manufactured_u_phys,
  make_affine_vertices,
  manufactured_u_phys,
  project_physical_laplacian_rhs,
  project_volume_coeffs_of_pullback,
)
from jlaplace import assemble_L_int  # noqa: E402


MANUFACTURED_DEGREE = 3


def compute_laplace_mansol_error(D, n, q_vol):
  """Return relative manufactured-solution error for the jlaplace L_int test."""
  if n < 2:
    raise ValueError("n must be at least 2 because L_int maps Pi_n to Pi_{n-2}")

  kappa = np.array([0.7 + 0.23 * i for i in range(D + 1)], dtype=np.float64)
  V_phys = make_affine_vertices(D)

  geom = affine_from_verts(V_phys)
  BinvT = geom["BinvT"]
  G = np.asfortranarray(BinvT.T @ BinvT)
  detBabs = float(geom["detBabs"])

  _, _, _, M = build_basis_structs(D, n, kappa)
  _, _, _, m_int = build_basis_structs(D, n - 2, kappa)

  Lij_ref = build_reference_second_partials(D, n, q_vol, kappa, kappa)
  L_int = assemble_L_int(D, G, detBabs, Lij_ref, m_int)

  c_u = project_volume_coeffs_of_pullback(
    D, n, kappa, q_vol, V_phys, manufactured_u_phys
  )
  lhs = L_int @ c_u

  rhs = project_physical_laplacian_rhs(
    D, n - 2, kappa, q_vol, V_phys, detBabs,
    laplacian_manufactured_u_phys,
  )

  rel = np.linalg.norm(lhs - rhs) / max(1e-300, np.linalg.norm(rhs))
  return float(rel), M, m_int, detBabs


def run_sweep(D, n_min, n_max, q_pad, q_fixed):
  rows = []

  for n in range(n_min, n_max + 1):
    if q_fixed is None:
      q_vol = n + q_pad
    else:
      q_vol = q_fixed

    rel, M, m_int, detBabs = compute_laplace_mansol_error(D, n, q_vol)

    row = {
      "n": n,
      "q_vol": q_vol,
      "M": M,
      "m_int": m_int,
      "detBabs": detBabs,
      "relerr": rel,
    }
    rows.append(row)

    print(
      f"D={D} n={n:2d} q_vol={q_vol:2d} M={M:5d} m_int={m_int:5d} "
      f"relerr={rel:.3e}"
    )

  return rows


def save_plot(rows, D, out_path):
  n_values = np.array([r["n"] for r in rows], dtype=np.int64)
  err_values = np.array([r["relerr"] for r in rows], dtype=np.float64)

  plt.figure(figsize=(7.0, 4.8))
  plt.semilogy(n_values, err_values, marker="o")
  plt.axvline(MANUFACTURED_DEGREE, linestyle="--", linewidth=1.0)
  plt.xlabel("Polynomial degree n")
  plt.ylabel("Relative manufactured-solution residual")
  plt.title(f"jlaplace manufactured-solution convergence, D={D}")
  plt.grid(True, which="both", linewidth=0.5)

  y_min = max(float(np.min(err_values)), 1e-15)
  plt.annotate(
    f"degree(u) = {MANUFACTURED_DEGREE}",
    xy=(MANUFACTURED_DEGREE, y_min),
    xytext=(MANUFACTURED_DEGREE + 0.3, max(y_min * 10.0, 1e-14)),
    arrowprops={"arrowstyle": "->"},
  )

  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  print(f"\nSaved plot to: {out_path}")


def main():
  parser = argparse.ArgumentParser(
    description="Sweep polynomial degree for the jlaplace manufactured-solution test."
  )
  parser.add_argument("--D", type=int, default=3, help="Simplex dimension.")
  parser.add_argument("--n-min", type=int, default=2, help="Minimum polynomial degree.")
  parser.add_argument("--n-max", type=int, default=9, help="Maximum polynomial degree.")
  parser.add_argument(
    "--q-pad",
    type=int,
    default=2,
    help="Use q_vol = n + q_pad when --q-fixed is omitted.",
  )
  parser.add_argument(
    "--q-fixed",
    type=int,
    default=None,
    help="Use a fixed q_vol instead of q_vol = n + q_pad.",
  )
  parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="Output PNG path. Defaults to testing/jlaplace_mansol_convergence_D{D}.png.",
  )
  parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Only print the convergence table; do not write a PNG.",
  )
  args = parser.parse_args()

  if args.n_min < 2:
    raise ValueError("--n-min must be at least 2 for L_int")
  if args.n_max < args.n_min:
    raise ValueError("--n-max must be >= --n-min")
  if args.q_pad < 0:
    raise ValueError("--q-pad must be nonnegative")

  rows = run_sweep(args.D, args.n_min, args.n_max, args.q_pad, args.q_fixed)

  print("\nSummary:")
  print("  Manufactured u_phys is degree 3, and affine pullback preserves degree.")
  print("  The error should drop to near roundoff once n >= 3, assuming q_vol is sufficient.")
  print("  Default quadrature is q_vol = n + q_pad, with q_pad = 2.")

  if not args.no_plot:
    out_path = args.out
    if out_path is None:
      out_path = os.path.join(_THIS_DIR, f"jlaplace_mansol_convergence_D{args.D}.png")
    save_plot(rows, args.D, out_path)


if __name__ == "__main__":
  main()
