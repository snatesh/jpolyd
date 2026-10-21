#!/usr/bin/env python3
"""Verify the reduced residual (Schur-complement) operator for a leaf.

Default: an illustrative, reproducible full-rank synthetic leaf.
Optional: --leaf-matrices saved.npz with L, T, tau (diagonal vector or SPD
matrix), and optionally s. Does not require libjpolyd or refactor an HPS tree.

The script checks both PSD and the exact reduced-energy/flux identities.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--leaf-matrices', type=Path, default=None)
  ap.add_argument('--output-dir', type=Path, default=Path('.'))
  args = ap.parse_args()
  out = args.output_dir
  out.mkdir(parents=True, exist_ok=True)

  if args.leaf_matrices is None:
    rng = np.random.default_rng(17)
    m, nr, nb = 15, 11, 19
    L = rng.standard_normal((nr, m))
    T = rng.standard_normal((nb, m))
    tau = 0.5 + rng.random(nb)
    s = 0.75
    label = 'Illustrative algebraic leaf (not measured JPolyD data)'
  else:
    z = np.load(args.leaf_matrices)
    L, T, tau = np.asarray(z['L'], float), np.asarray(z['T'], float), np.asarray(z['tau'], float)
    s = float(z['s']) if 's' in z else 1.0
    rng = np.random.default_rng(17)
    label = f'Exported leaf matrices: {args.leaf_matrices.name}'
  W = np.diag(tau) if tau.ndim == 1 else tau
  W = (W + W.T) / 2
  if L.ndim != 2 or T.ndim != 2 or L.shape[1] != T.shape[1] or W.shape != (T.shape[0], T.shape[0]):
    raise ValueError('Incompatible L, T, tau matrix dimensions')
  np.linalg.cholesky(W)
  H = (s*s) * (L.T @ L) + T.T @ W @ T
  np.linalg.cholesky(H)
  U = np.linalg.solve(H, T.T @ W)
  Vf = np.linalg.solve(H, (s*s) * L.T)
  P = W - W @ T @ U
  Ps = (P + P.T) / 2
  eigenvalues = np.linalg.eigvalsh(Ps)
  lam = rng.standard_normal(T.shape[0])
  f = rng.standard_normal(L.shape[0])
  a = U @ lam + Vf @ f
  dlt = T @ a - lam
  reduced_gradient = -W @ dlt
  expected_gradient = P @ lam - W @ T @ Vf @ f
  # The objective itself is the original stacked least-squares objective.
  objective = 0.5 * (s*s * np.linalg.norm(L @ a - f)**2 + dlt @ W @ dlt)
  a0 = U @ lam
  d0 = T @ a0 - lam
  min_energy = s*s * np.linalg.norm(L @ a0)**2 + d0 @ W @ d0
  quadratic_energy = lam @ P @ lam
  metrics = {
    'source': label,
    'shape_L': list(L.shape),
    'shape_T': list(T.shape),
    'min_eigenvalue_Psym': float(eigenvalues[0]),
    'max_eigenvalue_Psym': float(eigenvalues[-1]),
    'relative_symmetry_defect': float(np.linalg.norm(P-P.T) / max(np.linalg.norm(P), 1e-30)),
    'relative_gradient_identity_error': float(np.linalg.norm(reduced_gradient-expected_gradient) / max(np.linalg.norm(expected_gradient), 1e-30)),
    'relative_reduced_energy_identity_error': float(abs(min_energy-quadratic_energy) / max(abs(min_energy), 1e-30)),
    'objective_with_source': float(objective),
    'negative_eigenvalues_below_minus_1e_minus_9': int((eigenvalues < -1e-9).sum())
  }
  (out / 'pk_verification.json').write_text(json.dumps(metrics, indent=2) + '\n')
  fig, ax = plt.subplots(figsize=(7, 3.8))
  ax.plot(np.arange(1, len(eigenvalues)+1), eigenvalues, marker='o', markersize=3.5, linewidth=1)
  ax.axhline(0, linewidth=0.8)
  ax.set_xlabel('Eigenvalue index')
  ax.set_ylabel(r'Eigenvalue of $\operatorname{sym}(P_K)$')
  ax.set_title(r'Reduced residual operator $P_K = W-WTU_\lambda$')
  ax.grid(True, alpha=0.25)
  fig.tight_layout()
  fig.savefig(out / 'pk_psd_example.pdf')
  fig.savefig(out / 'pk_psd_example.png', dpi=160)
  plt.close(fig)
  print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
  main()
