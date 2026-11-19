"""Smooth nonpolynomial manufactured test: divergence-form Jacobi/HPS.

Uses the mesh, physical polynomial coefficients, variable Robin projection,
and weighted-L2 evaluation helpers from
  testing/test_hps_divergence_variable_robin_polynomial.py

The solution is an oscillatory exponential plus an interior Gaussian bump;
its gradient and Hessian are analytic. The physical source f = L u is
nonpolynomial and is projected into the degree-n residual Jacobi family.

Both HPS modes use identical data, with element balance measured but enforced
only in ElementBalance. The test checks the boundary/interface equations,
conservative element balance, source/Robin projection quadrature stability,
merge/root conditioning in both conservation modes, and the expected
reduction in error as n increases (unless already tiny).

Examples (from the repository root):
  python testing/test_hps_divergence_conservation_nonpoly.py \
    --D 1,2,3 --min-n 2 --max-n 7

  python testing/test_hps_divergence_conservation_nonpoly.py \
    --D 2 --min-n 3 --max-n 9 --kappas unweighted,chebyshev,asymmetric
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

from jhps import HpsDivergenceSolver, HpsLeafLeastSquaresSolver, HpsLeafOperatorMode
from jprecomp import RefSimplexPrecomp
from thread_control import set_omp_threads, set_openblas_threads

import test_hps_divergence_variable_robin_polynomial as reference


class SmoothManufactured(reference.Manufactured):
  """Analytic nonpolynomial solution with exact derivatives.

  The PDE coefficients A(x), b(x), c(x) and variable Robin alpha/beta
  are inherited unchanged from the polynomial manufactured regression.
  """

  def __init__(self, D: int):
    super().__init__(D)
    self.a = np.asarray([0.22 - 0.07 * j for j in range(D)])
    self.k = np.asarray([2.65 + 0.23 * j for j in range(D)])
    self.center = np.asarray([0.37 + 0.12 * (j % 2) for j in range(D)])
    self.bump_scale = 6.0
    self.wave_amplitude = 0.38
    self.bump_amplitude = 0.21
    self.variant_k = np.asarray([1.9 - 0.16 * j for j in range(D)])

  def u_grad_hess(self, X: np.ndarray, variant: int = 0):
    X = np.asarray(X, dtype=np.float64).reshape(-1, self.D)
    N, D = X.shape
    phase = 0.35 + X @ self.k
    ee = self.wave_amplitude * np.exp(X @ self.a)
    cc = np.cos(phase)
    ss = np.sin(phase)

    u = 1.30 + ee * cc
    grad = ee[:, None] * (
      cc[:, None] * self.a[None, :]
      - ss[:, None] * self.k[None, :]
    )
    aa_minus_kk = np.outer(self.a, self.a) - np.outer(self.k, self.k)
    ak_plus_ka = np.outer(self.a, self.k) + np.outer(self.k, self.a)
    hess = ee[:, None, None] * (
      cc[:, None, None] * aa_minus_kk[None, :, :]
      - ss[:, None, None] * ak_plus_ka[None, :, :]
    )

    r = X - self.center[None, :]
    bump = self.bump_amplitude * np.exp(
      -self.bump_scale * np.einsum('ki,ki->k', r, r)
    )
    u += bump
    grad += (-2.0 * self.bump_scale * bump)[:, None] * r
    hess += bump[:, None, None] * (
      4.0 * self.bump_scale ** 2 * r[:, :, None] * r[:, None, :]
      - 2.0 * self.bump_scale * np.eye(D)[None, :, :]
    )

    if variant:
      # Change the solution/source while keeping A,b,c unchanged.
      phase2 = X @ self.variant_k
      amp2 = 0.095
      u += amp2 * np.sin(phase2)
      grad += (amp2 * np.cos(phase2))[:, None] * self.variant_k
      hess -= (
        amp2 * np.sin(phase2)
      )[:, None, None] * np.outer(self.variant_k, self.variant_k)

    return u, grad, hess


def get_kappa(D: int, kind: str) -> np.ndarray:
  if kind == 'unweighted':
    return np.full(D + 1, 0.5)
  if kind == 'chebyshev':
    return np.zeros(D + 1)
  if kind == 'asymmetric':
    return np.asarray([0.71 + 0.17 * j for j in range(D + 1)])
  if kind == 'boundary_suppressed':
    return np.asarray([1.10 + 0.12 * j for j in range(D + 1)])
  raise ValueError(f'unknown kappa family: {kind}')


def balance_diagnostic(solver, source, enforce: bool, tol: float):
  row = solver.balance_residuals()
  assert row.shape == (solver.nelem,), row.shape
  assert np.all(np.isfinite(row)), row
  maximum = float(np.max(np.abs(row)))
  source_scale = max(1.0, float(np.max(np.abs(source))))
  if enforce:
    assert maximum < tol * source_scale, (maximum, tol, source_scale)
  return maximum


def assert_projection_stability(x, y, tol: float, label: str):
  scale = max(1.0, float(np.max(np.abs(y))))
  discrepancy = float(np.max(np.abs(x - y))) / scale
  assert discrepancy < tol, (label, discrepancy, tol)
  return discrepancy


MERGE_STRUCTURE_FIELDS = (
  'merge_index', 'parent_id', 'child_A_id', 'child_B_id', 'level',
  'interface_faces', 'interface_dofs', 'exterior_A_dofs',
  'exterior_B_dofs', 'parent_boundary_dofs',
  'child_A_boundary_dofs', 'child_B_boundary_dofs',
)


def inspect_merge_diagnostics(solver, merges, nelem, kf):
  """Check record integrity, including Schur-block partition dimensions."""
  records = solver.merge_condition_diagnostics()
  assert len(records) == len(merges)
  assert np.array_equal(records['merge_index'], np.arange(len(merges)))
  assert np.array_equal(records['parent_id'], nelem + np.arange(len(merges)))
  assert np.array_equal(records['child_A_id'], merges[:, 0])
  assert np.array_equal(records['child_B_id'], merges[:, 1])

  assert np.all(records['level'] >= 1)
  assert np.all(records['interface_faces'] >= 1)
  assert np.array_equal(
    records['interface_dofs'], records['interface_faces'] * kf
  )
  assert np.array_equal(
    records['child_A_boundary_dofs'],
    records['exterior_A_dofs'] + records['interface_dofs'],
  )
  assert np.array_equal(
    records['child_B_boundary_dofs'],
    records['exterior_B_dofs'] + records['interface_dofs'],
  )
  assert np.array_equal(
    records['parent_boundary_dofs'],
    records['exterior_A_dofs'] + records['exterior_B_dofs'],
  )

  for field in ('H_norm_1', 'H_rcond_1', 'parent_S_norm_1'):
    assert np.all(np.isfinite(records[field])), field
  assert np.all(records['H_norm_1'] > 0)
  assert np.all(records['parent_S_norm_1'] >= 0)
  assert np.all((records['H_rcond_1'] >= 0) &
                (records['H_rcond_1'] <= 1 + 1e-12))
  return records


def inspect_root_diagnostics(solver):
  """The root estimate describes the current Robin factorization."""
  root = solver.root_condition_diagnostics()
  assert root['dimension'] == solver.root_nb
  assert not root['singular']
  assert np.isfinite(root['matrix_norm_1'])
  assert root['matrix_norm_1'] > 0
  rcond = root['reciprocal_condition_1']
  assert np.isfinite(rcond) and 0 <= rcond <= 1 + 1e-12
  return root


def add_merge_records(output, records, D, family, n, mode):
  for record in records:
    row = dict(D=D, kappa=family, n=n, mode=mode)
    row.update({
      name: record[name].item() for name in records.dtype.names
    })
    output.append(row)


def compare_merge_modes(records_by_mode, D, family, n, comparison_rows):
  """Compare conditioning, but require equality only of tree structure."""
  none = records_by_mode['none']
  balanced = records_by_mode['element_balance']
  for name in MERGE_STRUCTURE_FIELDS:
    assert np.array_equal(none[name], balanced[name]), (D, family, n, name)

  tiny = np.finfo(float).tiny
  log_ratio = (
    np.log10(np.maximum(balanced['H_rcond_1'], tiny))
    - np.log10(np.maximum(none['H_rcond_1'], tiny))
  )
  for i, delta in enumerate(log_ratio):
    comparison_rows.append(dict(
      D=D, kappa=family, n=n,
      merge_index=int(none['merge_index'][i]),
      level=int(none['level'][i]),
      interface_dofs=int(none['interface_dofs'][i]),
      rcond_none=float(none['H_rcond_1'][i]),
      rcond_element_balance=float(balanced['H_rcond_1'][i]),
      delta_log10_rcond=float(delta),
    ))

  worst = int(np.argmax(np.abs(log_ratio)))
  print(
    f'    conditioning comparison: median log10(rcond_bal/rcond_none)='
    f'{np.median(log_ratio):+.2f}; largest shift merge='
    f'{int(none["merge_index"][worst])} level='
    f'{int(none["level"][worst])} '
    f'delta={log_ratio[worst]:+.2f} decades',
    flush=True,
  )


def write_csv(path: Path, records):
  if not records:
    return
  with path.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)


def plot_results(rows, output: Path):
  dimensions = sorted(set(row['D'] for row in rows))
  families = sorted(set(row['kappa'] for row in rows))
  for D in dimensions:
    for family in families:
      subset = [r for r in rows if r['D'] == D and r['kappa'] == family]
      if not subset:
        continue
      plots = [
        ('relative_L2', 'error', 'Relative weighted L2 error', 'Convergence'),
        ('balance_max', 'balance', 'Maximum element balance residual',
         'Physical elementwise balance'),
      ]
      if all('merge_min_rcond' in r for r in subset):
        plots.extend([
          ('merge_min_rcond', 'merge_rcond',
           'Minimum merge reciprocal condition', 'Worst interface conditioning'),
          ('root_rcond', 'root_rcond',
           'Root Robin reciprocal condition', 'Root conditioning'),
        ])
      for key, filename, ylabel, title in plots:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for mode in ('none', 'element_balance'):
          data = sorted((r for r in subset if r['mode'] == mode),
                        key=lambda r: r['n'])
          floor = (np.finfo(float).tiny if key in ('merge_min_rcond',
                                                  'root_rcond') else 1e-18)
          ax.semilogy([r['n'] for r in data],
                      [max(r[key], floor) for r in data],
                      marker='o', label=mode)
        if key == 'relative_L2':
          best = sorted((r for r in subset if r['mode'] == 'none'),
                        key=lambda r: r['n'])
          ax.semilogy([r['n'] for r in best],
                      [max(r['best_relative_L2'], 1e-18) for r in best],
                      marker='s', linestyle=':', label='Best weighted projection')
        ax.set_title(f'{title}: D={D}, kappa={family}')
        ax.set_xlabel('Polynomial degree n')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linewidth=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'nonpolynomial_D{D}_{family}_{filename}.png', dpi=180)
        plt.close(fig)


def run_dimension(D: int, family: str, args, rows, merge_rows, comparison_rows):
  ids, X, E, merges, keys, boundary, ninterior, cells = (
    reference.cube_simplicial_mesh(D, args.mesh_resolution)
  )
  model = SmoothManufactured(D)
  kappa = get_kappa(D, family)
  print(f'\nD={D} grid={cells} elements={len(E)} kappa={family} {kappa}',
        flush=True)

  convergence = {'none': [], 'element_balance': []}
  for n in range(args.min_n, args.max_n + 1):
    q_data = n + args.data_quad_extra
    pc = RefSimplexPrecomp(
      D, n, kappa,
      q_pad=args.solve_q_pad,
      q_vol=n + args.solve_q_pad,
      q_face=1 if D == 1 else n + args.solve_q_pad,
    )
    pc_data = RefSimplexPrecomp(
      D, n, kappa, q_pad=args.data_quad_extra,
      q_vol=q_data, q_face=1 if D == 1 else q_data,
    )
    pc_data_hi = RefSimplexPrecomp(
      D, n, kappa, q_pad=args.data_quad_extra + 4,
      q_vol=q_data + 4, q_face=1 if D == 1 else q_data + 4,
    )
    pc_eval = RefSimplexPrecomp(
      D, n, kappa, q_pad=args.data_quad_extra + 6,
      q_vol=q_data + 6, q_face=1 if D == 1 else q_data + 6,
    )

    A, b, c, f, u_projection = reference.project_volume(
      pc_data, X, E, model, n
    )
    alpha, beta, g, _ = reference.project_boundary(
      pc_data, X, E, boundary, model, n
    )
    _, _, _, f_hi, _ = reference.project_volume(
      pc_data_hi, X, E, model, n, include_coefficients=False
    )
    _, _, g_hi, _ = reference.project_boundary(
      pc_data_hi, X, E, boundary, model, n
    )
    f_quad_error = assert_projection_stability(
      f, f_hi, args.projection_tol, 'source'
    )
    g_quad_error = assert_projection_stability(
      g, g_hi, args.projection_tol, 'Robin'
    )
    assert np.max(np.abs(f)) > 1e-8, 'Manufactured source should be nonzero'

    records_by_mode = {}
    for mode in ('none', 'element_balance'):
      enforce = mode == 'element_balance'
      with HpsDivergenceSolver(
        pc, ids, X, E, merges, A, b, c, keys,
        p_A=n, p_b=n, p_c=n,
        assume_symmetric=True, tau_C=args.tau_C,
        alpha=1.0, beta=0.0,
        leaf_operator_mode=HpsLeafOperatorMode.DENSE,
        leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
        conservation_mode=mode,
        collect_merge_diagnostics=not args.no_merge_diagnostics,
      ) as solver:
        diagnostics = {}
        if not args.no_merge_diagnostics:
          records = inspect_merge_diagnostics(
            solver, merges, len(E), pc.kf
          )
          records_by_mode[mode] = records
          add_merge_records(merge_rows, records, D, family, n, mode)
          initial_root = inspect_root_diagnostics(solver)
          assert np.isclose(
            initial_root['reciprocal_condition_1'], 1.0, atol=1e-10
          ), initial_root
          diagnostics.update(
            merge_min_rcond=float(np.min(records['H_rcond_1'])),
            merge_median_rcond=float(np.median(records['H_rcond_1'])),
            merge_worst_index=int(records['merge_index'][
              np.argmin(records['H_rcond_1'])
            ]),
            merge_worst_level=int(records['level'][
              np.argmin(records['H_rcond_1'])
            ]),
            merge_max_interface_dofs=int(np.max(records['interface_dofs'])),
          )

        solver.update_variable_robin(alpha, beta, p_alpha=n, p_beta=n)
        if not args.no_merge_diagnostics:
          root = inspect_root_diagnostics(solver)
          assert np.array_equal(records, solver.merge_condition_diagnostics())
          diagnostics.update(
            root_rcond=float(root['reciprocal_condition_1']),
            root_matrix_norm_1=float(root['matrix_norm_1']),
          )

        result = solver.solve(f, g, check_residuals=True)
        reference.assert_result(
          result, len(E), len(boundary), ninterior, args.residual_tol
        )
        error, best = reference.relative_weighted_l2(
          pc_eval, X, E, result.leaf_coeffs, model
        )
        balance = balance_diagnostic(
          solver, f, enforce, args.balance_tol
        )
        convergence[mode].append(error)
        rows.append(dict(
          D=D, kappa=family, n=n, M=int(pc.M), mode=mode,
          elements=len(E), relative_L2=error, best_relative_L2=best,
          balance_max=balance,
          root_residual=result.root_robin_residual_inf,
          interface_residual=result.interface_flux_residual_inf,
          parent_residual=result.parent_consistency_residual_inf,
          source_quadrature_change=f_quad_error,
          robin_quadrature_change=g_quad_error,
          **diagnostics,
        ))
        print(f'  n={n:2d} M={pc.M:3d} mode={mode:15s} '
              f'relL2={error:.3e} best={best:.3e} '
              f'balance={balance:.3e} '
              f'root={result.root_robin_residual_inf:.2e} '
              f'iface={result.interface_flux_residual_inf:.2e}'
              + (f' min-rcond={diagnostics["merge_min_rcond"]:.2e} '
                 f'root-rcond={diagnostics["root_rcond"]:.2e}'
                 if diagnostics else ''), flush=True)

        if args.check_updates and n == args.max_n:
          # Same persistent tree, changed variable Robin coefficients.
          a2, b2, g2, _ = reference.project_boundary(
            pc_data, X, E, boundary, model, n, robin_variant=1
          )
          solver.update_variable_robin(a2, b2, p_alpha=n, p_beta=n)
          if not args.no_merge_diagnostics:
            updated_root = inspect_root_diagnostics(solver)
            assert np.array_equal(records, solver.merge_condition_diagnostics())
          changed_robin = solver.solve(f, g2, check_residuals=True)
          reference.assert_result(
            changed_robin, len(E), len(boundary), ninterior,
            args.residual_tol
          )
          robin_error, _ = reference.relative_weighted_l2(
            pc_eval, X, E, changed_robin.leaf_coeffs, model
          )
          robin_balance = balance_diagnostic(
            solver, f, enforce, args.balance_tol
          )

          # Change u and f, retaining PDE coefficients and Robin coefficients.
          _, _, _, f2, _ = reference.project_volume(
            pc_data, X, E, model, n, variant=1,
            include_coefficients=False
          )
          _, _, g3, _ = reference.project_boundary(
            pc_data, X, E, boundary, model, n,
            robin_variant=1, solution_variant=1
          )
          changed_source = solver.solve(f2, g3, check_residuals=True)
          reference.assert_result(
            changed_source, len(E), len(boundary), ninterior,
            args.residual_tol
          )
          source_error, _ = reference.relative_weighted_l2(
            pc_eval, X, E, changed_source.leaf_coeffs, model, variant=1
          )
          source_balance = balance_diagnostic(
            solver, f2, enforce, args.balance_tol
          )
          if not args.no_merge_diagnostics:
            assert inspect_root_diagnostics(solver) == updated_root
          print(f'    persistent updates [{mode}]: '
                f'robin L2={robin_error:.3e}, '
                f'source L2={source_error:.3e}, '
                f'balance Robin={robin_balance:.3e}, '
                f'balance source={source_balance:.3e}', flush=True)

    if not args.no_merge_diagnostics:
      compare_merge_modes(records_by_mode, D, family, n, comparison_rows)

  # Avoid imposing a particular rate, or a monotone decrease at every n.
  # A substantial endpoint decrease is expected for this smooth function.
  if args.max_n - args.min_n >= 2:
    for mode, errors in convergence.items():
      if errors[0] > 1e-10:
        assert errors[-1] < 0.5 * errors[0], (
          D, family, mode, 'no observed p-convergence', errors
        )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--D', default='1,2,3', help='Dimensions: subset of 1,2,3')
  parser.add_argument('--kappas', default='asymmetric',
                      help='Comma-separated: unweighted,chebyshev,asymmetric,'
                           'boundary_suppressed')
  parser.add_argument('--min-n', type=int, default=2)
  parser.add_argument('--max-n', type=int, default=7)
  parser.add_argument('--mesh-resolution', type=int, default=1)
  parser.add_argument('--solve-q-pad', type=int, default=2)
  parser.add_argument('--data-quad-extra', type=int, default=10)
  parser.add_argument('--tau-C', type=float, default=1.0)
  parser.add_argument('--omp-threads', type=int, default=4)
  parser.add_argument('--residual-tol', type=float, default=5e-8)
  parser.add_argument('--balance-tol', type=float, default=1e-8)
  parser.add_argument('--projection-tol', type=float, default=1e-9)
  parser.add_argument('--check-updates', action='store_true',
                      help='Check persistent Robin and source updates at max n')
  parser.add_argument('--no-merge-diagnostics', action='store_true',
                      help='Skip merge/root conditioning collection and comparisons')
  parser.add_argument('--output-dir', type=Path,
                      default=Path('build/hps_divergence_nonpolynomial'))
  args = parser.parse_args()

  dims = [int(v) for v in args.D.split(',') if v.strip()]
  kappas = [v.strip().lower() for v in args.kappas.split(',') if v.strip()]
  if not dims or any(D not in (1, 2, 3) for D in dims):
    parser.error('D must be a nonempty comma-separated subset of 1,2,3')
  if not kappas or any(k not in (
    'unweighted', 'chebyshev', 'asymmetric', 'boundary_suppressed'
  ) for k in kappas):
    parser.error('unrecognized kappa family')
  if args.min_n < 2 or args.max_n < args.min_n:
    parser.error('require 2 <= min-n <= max-n')
  if args.mesh_resolution < 1 or args.omp_threads < 1:
    parser.error('mesh resolution and omp threads must be positive')
  if args.data_quad_extra < 6:
    parser.error('data-quad-extra must be >= 6')

  set_openblas_threads(1)
  set_omp_threads(args.omp_threads)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  merge_rows = []
  comparison_rows = []
  for D in dims:
    for kappa in kappas:
      run_dimension(D, kappa, args, rows, merge_rows, comparison_rows)

  csv_path = args.output_dir / 'nonpolynomial_conservation.csv'
  write_csv(csv_path, rows)
  if not args.no_merge_diagnostics:
    write_csv(args.output_dir / 'nonpolynomial_merge_conditions.csv', merge_rows)
    write_csv(args.output_dir / 'nonpolynomial_merge_mode_comparison.csv',
              comparison_rows)
  plot_results(rows, args.output_dir)
  print('\nPASS: nonpolynomial divergence HPS, conservation and p-convergence '
        'checks', flush=True)
  print(f'CSV: {csv_path}', flush=True)
  if not args.no_merge_diagnostics:
    print(f'Merge diagnostics: '
          f'{args.output_dir / "nonpolynomial_merge_conditions.csv"}',
          flush=True)
    print(f'Mode comparison: '
          f'{args.output_dir / "nonpolynomial_merge_mode_comparison.csv"}',
          flush=True)
  print(f'Plots: {args.output_dir}/nonpolynomial_D*_*.png', flush=True)


if __name__ == '__main__':
  main()
