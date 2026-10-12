"""Nonzero physical boundary-mean gauge regression for persistent HPS.

Install the new jhps_elliptic_solver_set_neumann_gauge C/Python API, rebuild,
and put this file alongside:
  test_hps_divergence_pure_neumann.py
  test_hps_divergence_neumann_robin_transitions.py
  test_hps_divergence_neumann_high_conormal_transitions.py

Run:
  python testing/test_hps_divergence_neumann_nonzero_gauge.py --D 1,2,3 --n 4
  python testing/test_hps_divergence_neumann_nonzero_gauge.py --D 2,3 --n 5 --high-order

Checks:
  * changing eta in an already singular solver adds exactly eta to u;
  * the same RHS, outward co-normal flux and residuals are retained;
  * a nonfinite eta is rejected without changing the previous gauge;
  * a stored eta survives Robin -> variable-beta Neumann -> Robin transitions;
  * nonzero eta has no effect on a nonsingular positive-reaction flux solve;
  * optional cubic anisotropic diffusion with high-order variable-beta flux.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from jhps import (
  HpsDivergenceSolver,
  HpsLeafLeastSquaresSolver,
  HpsLeafOperatorMode,
)
from jprecomp import RefSimplexPrecomp, dimPi
from thread_control import set_omp_threads, set_openblas_threads
from test_hps_divergence_pure_neumann import (
  BASE_SUBDIVISIONS,
  assemble_data,
  check_diagnostics,
  cube_simplicial_mesh,
  physical_volume_points,
  solution_error,
)
from test_hps_divergence_neumann_robin_transitions import (
  project_boundary as simple_project_boundary,
)


def evaluate_difference(pc_eval, X, E, c1, c0, expected):
  """Compare u1-u0 directly on independently evaluated volume points."""
  Xhat, _ = pc_eval.volume_quad()
  V = pc_eval.volume_basis()
  max_difference = 0.0
  for e in range(len(E)):
    # Evaluation in physical coordinates is independent of the basis family.
    _P, _det = physical_volume_points(X[E[e]], Xhat)
    val = V @ (c1[e] - c0[e])
    max_difference = max(
      max_difference, float(np.max(np.abs(val - expected)))
    )
  return max_difference


def test_basic(D, n, args):
  ids, X, E, merges, keys, boundary, ni, cells = cube_simplicial_mesh(
    D, args.mesh_resolution
  )
  kappa = np.asarray([0.68 + 0.11 * j for j in range(D + 1)])
  pc = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.solve_q_pad,
    q_vol=n + args.solve_q_pad,
    q_face=1 if D == 1 else n + args.solve_q_pad,
  )
  pc_data = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.data_quad_extra,
    q_vol=n + args.data_quad_extra,
    q_face=1 if D == 1 else n + args.data_quad_extra,
  )
  pc_eval = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.data_quad_extra + 2,
    q_vol=n + args.data_quad_extra + 2,
    q_face=1 if D == 1 else n + args.data_quad_extra + 2,
  )
  A, _, _, f_zero, g_zero = assemble_data(
    pc_data, X, E, boundary, 'zero'
  )
  _, _, _, f_quad, g_quad = assemble_data(
    pc_data, X, E, boundary, 'quadratic'
  )
  a_zero, b_var, g_neu = simple_project_boundary(
    pc_data, X, E, boundary, 'zero', 'variable'
  )
  a_rob, b_rob, g_rob = simple_project_boundary(
    pc_data, X, E, boundary, 'robin', 'variable'
  )
  _, b_var2, g_neu2 = simple_project_boundary(
    pc_data, X, E, boundary, 'zero', 'variable2'
  )
  mean = (2 * D + 1) / 6
  opts = dict(
    p_A=0, alpha=0.0, beta=1.0, tau_C=args.tau_C,
    assume_symmetric=True,
    leaf_operator_mode=HpsLeafOperatorMode.DENSE,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
  )

  def check(result, label, eta, *, reaction=False):
    residual = check_diagnostics(
      result, len(E), len(boundary), ni, args.residual_tol
    )
    err, maxerr = solution_error(
      pc_eval, X, E, result.leaf_coeffs,
      'reaction' if reaction else 'quadratic',
      0.0 if reaction else mean - eta,
    )
    assert err < args.solution_tol, (D, label, 'relative L2', err)
    assert maxerr < args.solution_tol, (D, label, 'max error', maxerr)
    print(f'  {label:29s} relL2={err:.3e} max={maxerr:.3e} '
          f'residual={residual:.2e}', flush=True)
    return result

  print(f'\nD={D} n={n} cells={cells} nelem={len(E)} '
        f'faces={len(boundary)}', flush=True)

  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A, None, None, keys, **opts
  ) as solver:
    base = check(solver.solve(f_quad, g_quad), 'scalar Neumann eta=0', 0.0)
    old = base
    previous_eta = 0.0
    for eta in (0.75, -0.43, 1.3, 0.0):
      solver.set_neumann_gauge(eta)
      res = check(solver.solve(f_quad, g_quad),
                  f'scalar Neumann eta={eta:g}', eta)
      diff = evaluate_difference(
        pc_eval, X, E, res.leaf_coeffs, old.leaf_coeffs,
        eta - previous_eta,
      )
      assert diff < args.shift_tol, (D, eta, diff)
      old, previous_eta = res, eta

    solver.set_neumann_gauge(0.75)
    try:
      solver.set_neumann_gauge(float('nan'))
    except ValueError:
      pass
    else:
      raise AssertionError('NaN gauge was accepted')
    try:
      solver.set_neumann_gauge(float('inf'))
    except ValueError:
      pass
    else:
      raise AssertionError('infinite gauge was accepted')
    check(solver.solve(f_quad, g_quad), 'gauge retained after bad setter', 0.75)

    # With zero source and zero flux, the prescribed nonzero boundary mean
    # should select the constant solution u=eta, not u=0.
    zero = solver.solve(f_zero, g_zero)
    check_diagnostics(zero, len(E), len(boundary), ni, args.residual_tol)
    delta = evaluate_difference(
      pc_eval, X, E, zero.leaf_coeffs,
      np.zeros_like(zero.leaf_coeffs), 0.75,
    )
    assert delta < args.shift_tol, (D, 'zero f,g with eta=0.75', delta)
    print(f'  zero source/flux u=0.75          max_error={delta:.3e}', flush=True)

  # The left-nullspace compatibility check is nonsymmetric when b!=0.
  # Shifting a compatible solution by eta must still leave f and g unchanged.
  drift = np.asarray([0.23 + 0.07 * j for j in range(D)])
  A_drift, b_drift, _, f_drift, g_drift = assemble_data(
    pc_data, X, E, boundary, 'quadratic', drift=drift
  )
  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A_drift, b_drift, None, keys,
    **{**opts, 'p_b': 0}
  ) as solver:
    solver.set_neumann_gauge(0.38)
    check(solver.solve(f_drift, g_drift),
          'nonzero drift, eta=0.38', 0.38)

  # The new gauge must be dormant when a Robin condition fixes the constant.
  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A, None, None, keys,
    **{**opts, 'alpha': 1.0, 'beta': 0.0}
  ) as solver:
    solver.set_neumann_gauge(0.61)
    solver.update_variable_robin(
      a_zero, b_var, p_alpha=2, p_beta=2
    )
    check(solver.solve(f_quad, g_neu), 'Robin -> variable Neumann', 0.61)
    solver.set_neumann_gauge(-0.37)
    check(solver.solve(f_quad, g_neu), 'variable Neumann eta=-0.37', -0.37)

    solver.update_variable_robin(
      a_rob, b_rob, p_alpha=2, p_beta=2
    )
    # Under Robin, the exact manufactured solution is not shifted.
    check(solver.solve(f_quad, g_rob), 'Neumann -> variable Robin', mean)
    solver.set_neumann_gauge(0.94)
    check(solver.solve(f_quad, g_rob), 'Robin ignores stored eta', mean)
    solver.update_variable_robin(
      a_zero, b_var2, p_alpha=2, p_beta=2
    )
    check(solver.solve(f_quad, g_neu2), 'Robin -> Neumann eta=0.94', 0.94)

  # Nonsingular all-flux with c>0 also must ignore the stored gauge.
  A_reac, _, c, f_reac, g_reac = assemble_data(
    pc_data, X, E, boundary, 'reaction', reaction=0.6
  )
  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A_reac, None, c, keys,
    **{**opts, 'p_c': 0}
  ) as solver:
    solver.set_neumann_gauge(-2.0)
    check(solver.solve(f_reac, g_reac),
          'reaction: no gauge imposed', 0.0, reaction=True)


def test_high_order(D, n, args):
  """Cubic diffusion + variable beta, with nonzero eta on the full H_root."""
  from test_hps_divergence_neumann_high_conormal_transitions import (
    NeumannManufactured,
    boundary_mean_unit_cube,
    error_to_exact,
    project_boundary,
  )
  from test_hps_divergence_variable_robin_polynomial import (
    assert_result,
    project_volume,
  )
  ids, X, E, merges, keys, boundary, ni, cells = cube_simplicial_mesh(
    D, args.mesh_resolution
  )
  kappa = np.asarray([0.71 + 0.17 * j for j in range(D + 1)])
  qsolve = n + args.solve_q_pad
  qdata = n + args.data_quad_extra
  pc = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.solve_q_pad, q_vol=qsolve, q_face=qsolve
  )
  pc_data = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.data_quad_extra, q_vol=qdata, q_face=qdata
  )
  pc_eval = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.data_quad_extra + 2,
    q_vol=qdata + 2, q_face=qdata + 2,
  )
  model = NeumannManufactured(D)
  A_full, b_full, _, f, _ = project_volume(pc_data, X, E, model, n)
  A = np.ascontiguousarray(A_full[:, :, :, :dimPi(D, 3)])
  b = np.ascontiguousarray(b_full[:, :, :dimPi(D, 2)])
  a_zero, b_var, g = project_boundary(
    pc_data, X, E, boundary, model, 'neumann_a'
  )
  mean = boundary_mean_unit_cube(model)
  opts = dict(
    p_A=3, p_b=2, alpha=1.0, beta=0.0,
    leaf_operator_mode=HpsLeafOperatorMode.DENSE,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
    assume_symmetric=True, tau_C=args.tau_C,
  )
  print(f'\nHIGH ORDER D={D} n={n} nelem={len(E)} '
        f'boundary mean={mean:.9g}', flush=True)
  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A, b, None, keys, **opts
  ) as solver:
    solver.update_variable_robin(
      a_zero, b_var, p_alpha=2, p_beta=2
    )
    results = {}
    for eta in (0.0, 0.64, -0.29):
      solver.set_neumann_gauge(eta)
      res = solver.solve(f, g)
      assert_result(
        res, len(E), len(boundary), ni, args.residual_tol
      )
      rel, maxerr = error_to_exact(
        pc_eval, X, E, res.leaf_coeffs, model,
        shift=mean - eta,
      )
      assert rel < args.solution_tol, (D, n, eta, rel)
      assert maxerr < args.solution_tol, (D, n, eta, maxerr)
      results[eta] = res
      print(f'  variable high-order Neumann eta={eta:5.2f} '
            f'relL2={rel:.3e} max={maxerr:.3e}', flush=True)
    diff = evaluate_difference(
      pc_eval, X, E, results[0.64].leaf_coeffs,
      results[0.0].leaf_coeffs, 0.64,
    )
    assert diff < args.shift_tol, (D, n, 'high order constant shift', diff)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--D', default='1,2,3')
  parser.add_argument('--n', type=int, default=4)
  parser.add_argument('--high-order', action='store_true')
  parser.add_argument('--mesh-resolution', type=int, default=1)
  parser.add_argument('--solve-q-pad', type=int, default=2)
  parser.add_argument('--data-quad-extra', type=int, default=7)
  parser.add_argument('--tau-C', type=float, default=1.0)
  parser.add_argument('--omp-threads', type=int, default=4)
  parser.add_argument('--solution-tol', type=float, default=2e-8)
  parser.add_argument('--residual-tol', type=float, default=5e-8)
  parser.add_argument('--shift-tol', type=float, default=5e-9)
  args = parser.parse_args()
  dims = [int(t) for t in args.D.split(',') if t.strip()]
  if not dims or any(d not in BASE_SUBDIVISIONS for d in dims):
    parser.error('--D must be a subset of 1,2,3')
  if args.n < 3:
    parser.error('require n>=3')
  if args.high_order and (args.n < 5 or any(d == 1 for d in dims)):
    parser.error('--high-order requires n>=5 and D from 2,3')
  set_openblas_threads(1)
  set_omp_threads(args.omp_threads)
  for D in dims:
    if args.high_order:
      test_high_order(D, args.n, args)
    else:
      test_basic(D, args.n, args)
  print(f'\nPASS: nonzero Neumann gauge D={dims}, n={args.n}'
        f'{" high-order" if args.high_order else ""}', flush=True)


if __name__ == '__main__':
  main()
