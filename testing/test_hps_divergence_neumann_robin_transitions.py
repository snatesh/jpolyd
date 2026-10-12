"""Persistent divergence HPS: transactional Robin <-> Neumann updates.

Requires test_hps_divergence_pure_neumann.py in the same testing directory.
Run from your usual jpolyd Python environment:

  python testing/test_hps_divergence_neumann_robin_transitions.py --D 1,2,3 --n 4

Checks:
  * initially Dirichlet -> variable-beta pure Neumann with zero mean gauge;
  * compatible and incompatible source/flux, including reuse after failure;
  * variable Robin -> different variable-beta Neumann -> Robin;
  * invalid all-zero alpha/beta update rejected without changing a valid
    bordered or LU root factorization;
  * with positive reaction, variable-beta pure flux remains nonsingular.

The manufactured solution is u=sum(x_j^2). Under the singular pure-flux
boundary condition the expected solution is shifted by the exact physical
boundary mean (2D+1)/6. Under Robin/Dirichlet, u is unshifted.
"""
from __future__ import annotations

import argparse
import numpy as np

from jhps import HpsDivergenceSolver, HpsLeafLeastSquaresSolver, HpsLeafOperatorMode
from jprecomp import RefSimplexPrecomp, dimPi
from thread_control import set_omp_threads, set_openblas_threads

from test_hps_divergence_pure_neumann import (
  BASE_SUBDIVISIONS,
  assemble_data,
  check_diagnostics,
  cube_simplicial_mesh,
  face_points_normal_scale,
  solution_error,
)


def boundary_fields(P, alpha_kind, beta_kind):
  """Physical boundary alpha,beta; all exactly degree <= 2."""
  ss = P.sum(axis=1)
  ss2 = (P * P).sum(axis=1)

  if alpha_kind == 'zero':
    alpha = np.zeros(len(P))
  elif alpha_kind == 'dirichlet':
    alpha = np.ones(len(P))
  elif alpha_kind == 'robin':
    alpha = 0.85 + 0.13 * ss
  elif alpha_kind == 'robin2':
    alpha = 1.22 + 0.07 * ss2
  else:
    raise ValueError(alpha_kind)

  if beta_kind == 'zero':
    beta = np.zeros(len(P))
  elif beta_kind == 'one':
    beta = np.ones(len(P))
  elif beta_kind == 'variable':
    beta = 1.1 + 0.14 * P[:, 0] + 0.09 * ss2
  elif beta_kind == 'variable2':
    beta = 1.35 + 0.075 * ss + 0.06 * P[:, 0]**2
  else:
    raise ValueError(beta_kind)
  return alpha, beta


def project_boundary(pc_data, X, E, boundary, alpha_kind, beta_kind,
                     *, solution_kind='quadratic'):
  """Root-face ordering, unscaled Robin-coefficient modes, scaled g moments."""
  D = pc_data.D
  Y, weights, V = pc_data.face_quad_basis()
  p = 2  # Fixed coefficient degree in both alpha and beta.
  kp = dimPi(D - 1, p)
  va = np.empty((len(boundary), kp))
  vb = np.empty((len(boundary), kp))
  g = np.empty((len(boundary), pc_data.kf))
  for row, (_key, (e, face_id)) in enumerate(boundary):
    P, normal, scale = face_points_normal_scale(
      D, X[E[e]], E[e], face_id, Y
    )
    alpha, beta = boundary_fields(P, alpha_kind, beta_kind)
    if solution_kind == 'quadratic':
      u = np.sum(P * P, axis=1)
      flux = 2 * (P @ normal)
    elif solution_kind == 'constant':
      u = np.ones(len(P))
      flux = np.zeros(len(P))
    elif solution_kind == 'incompatible':
      u = np.zeros(len(P))
      flux = np.ones(len(P))
    else:
      raise ValueError(solution_kind)

    va[row] = V[:, :kp].T @ (weights * alpha)
    vb[row] = V[:, :kp].T @ (weights * beta)
    g[row] = scale * (V.T @ (weights * (alpha * u + beta * flux)))
  return (np.ascontiguousarray(va), np.ascontiguousarray(vb),
          np.ascontiguousarray(g))


def expect_update_failure(solver, alpha_modes, beta_modes):
  try:
    solver.update_variable_robin(alpha_modes, beta_modes, p_alpha=2, p_beta=2)
  except RuntimeError as exc:
    print(f'  invalid update rejected: {exc}', flush=True)
    return
  raise AssertionError('all-zero alpha/beta update should have failed')


def run_dimension(D, args):
  ids, X, E, merges, keys, boundary, ni, cells = cube_simplicial_mesh(
    D, args.mesh_resolution
  )
  n = args.n
  kappa = np.asarray([0.68 + 0.11 * j for j in range(D + 1)])
  pc = RefSimplexPrecomp(
    D, n, kappa,
    q_pad=args.solve_q_pad, q_vol=n + args.solve_q_pad,
    q_face=1 if D == 1 else n + args.solve_q_pad,
  )
  pc_data = RefSimplexPrecomp(
    D, n, kappa,
    q_pad=args.data_quad_extra, q_vol=n + args.data_quad_extra,
    q_face=1 if D == 1 else n + args.data_quad_extra,
  )
  pc_eval = RefSimplexPrecomp(
    D, n, kappa,
    q_pad=args.data_quad_extra + 2, q_vol=n + args.data_quad_extra + 2,
    q_face=1 if D == 1 else n + args.data_quad_extra + 2,
  )

  A, _, _, f_zero, g_zero = assemble_data(
    pc_data, X, E, boundary, 'zero'
  )
  _, _, _, f_quad, _ = assemble_data(
    pc_data, X, E, boundary, 'quadratic'
  )
  a_dir, b_zero, g_dir = project_boundary(
    pc_data, X, E, boundary, 'dirichlet', 'zero'
  )
  a_zero, b_var, g_neu = project_boundary(
    pc_data, X, E, boundary, 'zero', 'variable'
  )
  _, b_var2, g_neu2 = project_boundary(
    pc_data, X, E, boundary, 'zero', 'variable2'
  )
  a_rob, b_rob, g_rob = project_boundary(
    pc_data, X, E, boundary, 'robin', 'variable'
  )
  a_rob2, b_rob2, g_rob2 = project_boundary(
    pc_data, X, E, boundary, 'robin2', 'variable2'
  )
  _, _, g_bad = project_boundary(
    pc_data, X, E, boundary, 'zero', 'variable',
    solution_kind='incompatible'
  )
  mean = (2.0 * D + 1.0) / 6.0

  print(f'\nD={D}, n={n}, cells={cells}, elements={len(E)}, '
        f'boundary_faces={len(boundary)}, interfaces={ni}', flush=True)

  kwargs = dict(
    p_A=0, assume_symmetric=True, tau_C=args.tau_C,
    leaf_operator_mode=HpsLeafOperatorMode.DENSE,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
  )

  def verify(solver, f, g, expected, label, *, shifted=False):
    result = solver.solve(f, g, check_residuals=True)
    residual = check_diagnostics(
      result, len(E), len(boundary), ni, args.residual_tol
    )
    l2, maximum = solution_error(
      pc_eval, X, E, result.leaf_coeffs, expected,
      mean if shifted else 0.0
    )
    assert l2 < args.solution_tol, (D, label, 'L2', l2)
    assert maximum < args.solution_tol, (D, label, 'max', maximum)
    print(f'  {label:32s} L2={l2:.3e} max={maximum:.3e} '
          f'residual={residual:.2e}', flush=True)

  def update(solver, a, b):
    solver.update_variable_robin(a, b, p_alpha=2, p_beta=2)

  # Initially nonsingular scalar Dirichlet; requires eager gauge construction
  # to transition to Neumann later without reconstructing the HPS tree.
  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A, None, None, keys,
    alpha=1.0, beta=0.0, **kwargs
  ) as solver:
    verify(solver, f_quad, g_dir, 'quadratic', 'initial scalar Dirichlet')

    update(solver, a_zero, b_var)
    verify(solver, f_quad, g_neu, 'quadratic',
           'Dirichlet -> variable Neumann', shifted=True)

    # Failure during a solve must preserve the bordered factorization.
    try:
      solver.solve(f_zero, g_bad)
    except RuntimeError as exc:
      print(f'  incompatible Neumann rejected: {exc}', flush=True)
    else:
      raise AssertionError('incompatible Neumann RHS was accepted')
    verify(solver, f_quad, g_neu, 'quadratic',
           'Neumann after bad RHS', shifted=True)

    # Invalid all-zero alpha/beta should be rejected before state commit.
    expect_update_failure(solver, a_zero, b_zero)
    verify(solver, f_quad, g_neu, 'quadratic',
           'Neumann after rejected update', shifted=True)

    update(solver, a_rob, b_rob)
    verify(solver, f_quad, g_rob, 'quadratic',
           'variable Neumann -> Robin')

    expect_update_failure(solver, a_zero, b_zero)
    verify(solver, f_quad, g_rob, 'quadratic',
           'Robin after rejected update')

    update(solver, a_zero, b_var2)
    verify(solver, f_quad, g_neu2, 'quadratic',
           'Robin -> changed variable Neumann', shifted=True)

    update(solver, a_rob2, b_rob2)
    verify(solver, f_quad, g_rob2, 'quadratic',
           'changed Neumann -> changed Robin')

  # With positive reaction, pure-flux boundary data are nonsingular.
  # The Neumann update must NOT impose the zero boundary-mean gauge.
  if not args.skip_reaction:
    reaction = 0.6
    A_react, _, c_react, f_react, _ = assemble_data(
      pc_data, X, E, boundary, 'reaction', reaction=reaction
    )
    _, _, g_react = project_boundary(
      pc_data, X, E, boundary, 'zero', 'variable',
      solution_kind='constant'
    )
    with HpsDivergenceSolver(
      pc, ids, X, E, merges, A_react, None, c_react, keys,
      p_c=0, alpha=1.0, beta=0.0, **kwargs
    ) as solver:
      update(solver, a_zero, b_var)
      verify(solver, f_react, g_react, 'reaction',
             'reaction: updated pure-flux LU')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--D', default='1,2,3')
  parser.add_argument('--n', type=int, default=4)
  parser.add_argument('--mesh-resolution', type=int, default=1)
  parser.add_argument('--solve-q-pad', type=int, default=2)
  parser.add_argument('--data-quad-extra', type=int, default=7)
  parser.add_argument('--tau-C', type=float, default=1.0)
  parser.add_argument('--omp-threads', type=int, default=4)
  parser.add_argument('--solution-tol', type=float, default=2e-8)
  parser.add_argument('--residual-tol', type=float, default=5e-8)
  parser.add_argument('--skip-reaction', action='store_true')
  args = parser.parse_args()
  dims = [int(d) for d in args.D.split(',') if d.strip()]
  if not dims or any(D not in BASE_SUBDIVISIONS for D in dims):
    parser.error('D must be a comma-separated subset of 1,2,3')
  if args.n < 3 or args.mesh_resolution < 1:
    parser.error('require n>=3 and mesh-resolution>=1')
  if args.solve_q_pad < 0 or args.data_quad_extra < 5 or args.omp_threads < 1:
    parser.error('invalid quadrature or thread options')
  set_openblas_threads(1)
  set_omp_threads(args.omp_threads)
  for D in dims:
    run_dimension(D, args)
  print(f'\nPASS: variable Robin <-> Neumann transitions D={dims}, n={args.n}',
        flush=True)


if __name__ == '__main__':
  main()
