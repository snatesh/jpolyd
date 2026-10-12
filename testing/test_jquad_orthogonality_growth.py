"""Jacobi simplex quadrature orthogonality defect versus polynomial degree.

Uses the existing jquad_tprod and jbasis Python bindings, with NO HPS solve.
The quadrature and basis are defined for the same weighted Jacobi measure.

Examples:
  python testing/test_jquad_orthogonality_growth.py
  python testing/test_jquad_orthogonality_growth.py \
    --dims 2,3 --min-n 2 --max-n 14 --q-extras 1,3,7

The default kappa matches the variable-Robin polynomial regression:
  kappa_D[j] = 0.71 + 0.17*j, j=0,...,D.
The residual family uses kappa_D + 2 in every component. The canonical
face family uses the first D entries of kappa_D, in dimension D-1.

Output: CSV and two PNG plots in build/quadrature_orthogonality/.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from jbasis import jbasis_build_structures, jbasis_eval_all
from jquad_tprod import jquad_mapped_build_kappa


def comma_ints(value: str) -> list[int]:
  try:
    values = [int(x.strip()) for x in value.split(',')]
  except ValueError as exc:
    raise argparse.ArgumentTypeError('expected comma-separated integers') from exc
  if not values:
    raise argparse.ArgumentTypeError('expected at least one integer')
  return values


def comma_words(value: str) -> list[str]:
  return [x.strip() for x in value.split(',') if x.strip()]


def gram_defect(D: int, n: int, q: int, kappa: np.ndarray,
                structures: tuple) -> dict:
  """Return defects of G=V.T diag(w) V, with all measures weighted identically."""
  points, weights = jquad_mapped_build_kappa(D, q, kappa)
  alpha_table, tail_deg, inv_h = structures
  V = jbasis_eval_all(points, kappa, n, alpha_table, tail_deg, inv_h, D)
  M = V.shape[1]

  G = V.T @ (weights[:, None] * V)
  # The exact Gram matrix is symmetric. Measure the asymmetry separately;
  # eigvalsh then computes the spectral norm of the symmetrized defect.
  asym = float(np.max(np.abs(G - G.T)))
  E = 0.5 * (G + G.T)
  E.flat[::M + 1] -= 1.0

  eigenvalues = np.linalg.eigvalsh(E)
  diagonal = np.diag(E)
  # Avoid building a full off-diagonal copy.
  max_entry = float(np.max(np.abs(E)))
  max_offdiag = float(np.max(np.abs(E - np.diag(diagonal)))) if M > 1 else 0.0

  return {
    'M': M,
    'npoints': len(weights),
    'spectral': float(np.max(np.abs(eigenvalues))),
    'frobenius': float(np.linalg.norm(E, ord='fro')),
    'max_entry': max_entry,
    'max_diag': float(np.max(np.abs(diagonal))),
    'max_offdiag': max_offdiag,
    'asym': asym,
    'weight_sum': float(np.sum(weights)),
  }


def families_for_dim(D: int, selected: list[str]):
  base = np.array([0.71 + 0.17 * j for j in range(D + 1)], dtype=float)
  all_families = {
    'volume': (D, base),
    'residual': (D, base + 2.0),
    'face': (D - 1, base[:D].copy()),
  }
  return {name: all_families[name] for name in selected}


def save_plot(rows: list[dict], dimensions: list[int], metric: str,
              filename: Path):
  fig, axes = plt.subplots(1, len(dimensions), figsize=(6.5 * len(dimensions), 4.7),
                           sharey=True, squeeze=False)
  for ax, D in zip(axes[0], dimensions):
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
      if row['D'] == D:
        groups.setdefault((row['family'], row['q_extra']), []).append(row)
    for (family, q_extra), group in sorted(groups.items()):
      group.sort(key=lambda r: r['n'])
      ax.semilogy([r['n'] for r in group],
                  [max(r[metric], np.finfo(float).tiny) for r in group],
                  '-o', ms=3, label=f'{family}, q=n+{q_extra}')
    ax.set_title(f'Volume dimension D={D}')
    ax.set_xlabel('Jacobi polynomial degree n')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
  axes[0, 0].set_ylabel({
    'spectral': r'$\|\operatorname{sym}(V^TWV)-I\|_2$',
    'frobenius': r'$\|\operatorname{sym}(V^TWV)-I\|_F$',
  }[metric])
  fig.suptitle('Mapped tensor-product Jacobi quadrature: orthogonality defect')
  fig.tight_layout()
  fig.savefig(filename, dpi=180)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--dims', type=comma_ints, default=[2, 3],
                      help='Volume dimensions; default 2,3')
  parser.add_argument('--min-n', type=int, default=2)
  parser.add_argument('--max-n', type=int, default=12)
  parser.add_argument('--q-extras', type=comma_ints, default=[1, 3, 7],
                      help='1D tensor quadrature orders q=n+extra; default 1,3,7')
  parser.add_argument('--families', type=comma_words,
                      default=['volume', 'residual', 'face'],
                      help='Subset of volume,residual,face')
  parser.add_argument('--output-dir', type=Path,
                      default=Path('build/quadrature_orthogonality'))
  args = parser.parse_args()

  if not args.dims or any(D not in (2, 3) for D in args.dims):
    parser.error('--dims must be drawn from 2,3')
  if args.min_n < 0 or args.max_n < args.min_n:
    parser.error('require 0 <= min-n <= max-n')
  if any(v < 1 for v in args.q_extras):
    parser.error('--q-extras values must be >= 1, giving q >= n+1')
  if any(name not in ('volume', 'residual', 'face') for name in args.families):
    parser.error('--families must be drawn from volume,residual,face')

  args.output_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for D in args.dims:
    for family, (d, kappa) in families_for_dim(D, args.families).items():
      print(f'\nD={D}, family={family}, basis_dimension={d}, '
            f'kappa={np.array2string(kappa, precision=3)}', flush=True)
      print('   n    M    q   nodes     ||E||_2     ||E||_F'
            '     max|E|     max|diag|    max|offdiag|', flush=True)
      for n in range(args.min_n, args.max_n + 1):
        # Independent of quadrature order; build only once per degree/family.
        structures = jbasis_build_structures(d, n, kappa)
        for extra in args.q_extras:
          q = n + extra
          result = gram_defect(d, n, q, kappa, structures)
          row = dict(D=D, family=family, basis_D=d, n=n,
                     q=q, q_extra=extra, **result)
          rows.append(row)
          print(f'{n:4d} {result["M"]:4d} {q:4d} {result["npoints"]:7d}'
                f' {result["spectral"]:11.3e}'
                f' {result["frobenius"]:11.3e}'
                f' {result["max_entry"]:11.3e}'
                f' {result["max_diag"]:11.3e}'
                f' {result["max_offdiag"]:11.3e}', flush=True)

  if not rows:
    return
  csv_path = args.output_dir / 'jacobi_gram_defects.csv'
  with csv_path.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  spectral_path = args.output_dir / 'jacobi_gram_spectral.png'
  frobenius_path = args.output_dir / 'jacobi_gram_frobenius.png'
  save_plot(rows, args.dims, 'spectral', spectral_path)
  save_plot(rows, args.dims, 'frobenius', frobenius_path)
  print(f'\nCSV: {csv_path}\nSpectral plot: {spectral_path}'
        f'\nFrobenius plot: {frobenius_path}')


if __name__ == '__main__':
  main()
