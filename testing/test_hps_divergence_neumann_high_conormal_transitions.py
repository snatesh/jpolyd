"""High-order co-normal regression for Robin <-> Neumann HPS transitions.

Put alongside test_hps_divergence_variable_robin_polynomial.py in testing/.
Uses its established Kuhn meshes, cubic anisotropic SPD tensor and polynomial
projection machinery; sets c=0 to make the pure-flux root singular.

Run:
  python testing/test_hps_divergence_neumann_high_conormal_transitions.py \\
    --D 2,3 --n 5,6

The degree-5 exact solution has modestly amplified fifth-degree terms so the
physical flux's high-face modes, and their coupling through variable beta,
are measurable. A is still the original degree-3 anisotropic SPD tensor;
b is the original degree-2 drift. Exact solution is shifted by its *physical*
boundary mean in singular Neumann stages, and unshifted in Robin stages.
"""
from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

from jhps import (
  HpsDivergenceSolver,
  HpsLeafLeastSquaresSolver,
  HpsLeafOperatorMode,
)
from jprecomp import RefSimplexPrecomp, dimPi
from thread_control import set_omp_threads, set_openblas_threads

from test_hps_divergence_variable_robin_polynomial import (
  Manufactured,
  assert_result,
  cube_simplicial_mesh,
  face_points_normal_scale,
  project_volume,
  volume_points,
)


class NeumannManufactured(Manufactured):
  """Validated cubic A and quadratic b, but c=0 and stronger degree-5 modes."""

  def __init__(self, D):
    super().__init__(D)
    self.epsilon = 3.0
    # Strengthen the degree-5 content specifically to test M_beta^{LH} H_root.
    for j in range(D):
      power = [0] * D
      power[j] = 5
      self.terms.append((0.30 + 0.08 * j, tuple(power)))

  def coefficients(self, X):
    A, divA, b, _ = super().coefficients(X)
    return A, divA, b, np.zeros(len(X))


def boundary_mean_unit_cube(model, *, variant=0, order=12):
  """Independent unweighted integral over the 2D unit-cube boundary faces.

  A degree-5 polynomial is integrated exactly (up to floating-point errors)
  using one-dimensional Gauss-Legendre tensor-product quadrature.
  """
  D = model.D
  t, w = np.polynomial.legendre.leggauss(order)
  t = (t + 1.0) / 2.0
  w = w / 2.0
  other = list(itertools.product(range(order), repeat=D - 1))
  integ = 0.0
  for fixed_axis in range(D):
    for boundary_coordinate in (0.0, 1.0):
      P = np.zeros((len(other), D))
      P[:, fixed_axis] = boundary_coordinate
      W = np.ones(len(other))
      for row, multi in enumerate(other):
        for i, axis in enumerate(j for j in range(D) if j != fixed_axis):
          P[row, axis] = t[multi[i]]
          W[row] *= w[multi[i]]
      integ += float(W @ model.u_grad_hess(P, variant)[0])
  # Each of the 2D faces has physical measure 1.
  return integ / (2.0 * D)


def boundary_values(model, P, condition):
  """Polynomial Robin coefficients of degree <= 2."""
  if condition == 'dirichlet':
    return np.ones(len(P)), np.zeros(len(P))
  if condition == 'neumann_a':
    alpha = np.zeros(len(P))
    beta = 1.10 + 0.16 * P[:, 0] + 0.09 * np.sum(P * P, axis=1)
    return alpha, beta
  if condition == 'neumann_b':
    alpha = np.zeros(len(P))
    beta = 1.32 + 0.11 * np.sum(P, axis=1) + 0.13 * P[:, -1] ** 2
    return alpha, beta
  if condition == 'robin_a':
    return model.robin(P, 0)
  if condition == 'robin_b':
    return model.robin(P, 1)
  if condition == 'zero':
    return np.zeros(len(P)), np.zeros(len(P))
  raise ValueError(condition)


def project_boundary(pc_data, X, E, boundary, model, condition, *, variant=0,
                     add_incompatible_flux=0.0):
  """Project scalar coefficients and boundary data in input face order.

  Robin alpha and beta: unscaled modal coefficients, base face kappa family.
  g: physical-face-scaled modal moments, including the true co-normal flux.
  """
  D = model.D
  Y, w, V = pc_data.face_quad_basis()
  ka = dimPi(D - 1, 2)
  Va = V[:, :ka]
  a = np.empty((len(boundary), ka))
  b = np.empty_like(a)
  g = np.empty((len(boundary), pc_data.kf))
  for row, (_key, (e, face_id)) in enumerate(boundary):
    P, normal, scale = face_points_normal_scale(D, X[E[e]], E[e], face_id, Y)
    u, grad, _ = model.u_grad_hess(P, variant)
    A, _, _, _ = model.coefficients(P)
    q = np.einsum('i,kij,kj->k', normal, A, grad)
    alpha, beta = boundary_values(model, P, condition)
    a[row] = Va.T @ (w * alpha)
    b[row] = Va.T @ (w * beta)
    # Deliberately positive added physical boundary RHS tests compatibility.
    g_phys = alpha * u + beta * q + add_incompatible_flux
    g[row] = scale * (V.T @ (w * g_phys))
  return tuple(np.ascontiguousarray(v) for v in (a, b, g))


def high_flux_coupling_inf(pc_hi, X, E, boundary, model, n, condition):
  """Independently measure M_beta^{LH} q_H on exterior faces.

  q_H is the part of physical co-normal flux orthogonal to face Pi_n.
  Using a face basis through degree n+2 (the true degree of A grad u),
  we project beta * q_H back to Pi_n. This should not vanish for the
  manufactured diffusion, solution and variable beta.
  """
  Y, w, V = pc_hi.face_quad_basis()
  kf = dimPi(model.D - 1, n)
  VL = V[:, :kf]
  highest = 0.0
  for _key, (e, face_id) in boundary:
    P, normal, scale = face_points_normal_scale(
      model.D, X[E[e]], E[e], face_id, Y
    )
    _, grad, _ = model.u_grad_hess(P)
    A, _, _, _ = model.coefficients(P)
    q = np.einsum('i,kij,kj->k', normal, A, grad)
    _, beta = boundary_values(model, P, condition)
    q_low = VL @ (VL.T @ (w * q))
    correction = scale * (VL.T @ (w * beta * (q - q_low)))
    highest = max(highest, float(np.max(np.abs(correction))))
  return highest


def error_to_exact(pc_eval, X, E, coeffs, model, *, variant=0, shift=0.0):
  Xref, w = pc_eval.volume_quad()
  V = pc_eval.volume_basis()
  err = ref = 0.0
  maximum = 0.0
  for e, elem in enumerate(E):
    P, jac = volume_points(X[elem], Xref)
    u = model.u_grad_hess(P, variant)[0] - shift
    uh = V @ coeffs[e]
    diff = uh - u
    err += jac * float(w @ (diff * diff))
    ref += jac * float(w @ (u * u))
    maximum = max(maximum, float(np.max(np.abs(diff))))
  return math.sqrt(err / max(ref, 1e-300)), maximum


def run_one(D, n, args):
  ids, X, E, merges, keys, boundary, ninterior, cells = cube_simplicial_mesh(
    D, args.mesh_resolution
  )
  kappa = np.asarray([0.71 + 0.17 * j for j in range(D + 1)])
  qsolve = n + args.solve_q_pad
  qdata = n + args.data_quad_extra
  pc = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.solve_q_pad,
    q_vol=qsolve, q_face=qsolve,
  )
  pc_data = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.data_quad_extra,
    q_vol=qdata, q_face=qdata,
  )
  pc_eval = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.data_quad_extra + 2,
    q_vol=qdata + 2, q_face=qdata + 2,
  )
  pc_hi = RefSimplexPrecomp(
    D, n + 2, kappa, q_pad=args.data_quad_extra,
    q_vol=qdata + 2, q_face=qdata + 2,
  )
  model = NeumannManufactured(D)
  p_A, p_b = 3, 2
  full_A, full_b, _, f0, _ = project_volume(pc_data, X, E, model, n)
  A = np.ascontiguousarray(full_A[:, :, :, :dimPi(D, p_A)])
  b = np.ascontiguousarray(full_b[:, :, :dimPi(D, p_b)])
  _, _, _, f1, _ = project_volume(
    pc_data, X, E, model, n, variant=1, include_coefficients=False
  )
  a_dir, b_dir, g_dir = project_boundary(
    pc_data, X, E, boundary, model, 'dirichlet'
  )
  a_neu, b_neu, g_neu = project_boundary(
    pc_data, X, E, boundary, model, 'neumann_a'
  )
  _, b_neu2, g_neu2 = project_boundary(
    pc_data, X, E, boundary, model, 'neumann_b', variant=1
  )
  a_rob, b_rob, g_rob = project_boundary(
    pc_data, X, E, boundary, model, 'robin_a'
  )
  a_rob2, b_rob2, g_rob2 = project_boundary(
    pc_data, X, E, boundary, model, 'robin_b', variant=1
  )
  a_bad, b_bad, _ = project_boundary(
    pc_data, X, E, boundary, model, 'zero'
  )
  _, _, g_incompatible = project_boundary(
    pc_data, X, E, boundary, model, 'neumann_a',
    add_incompatible_flux=args.incompatible_offset,
  )
  mean0 = boundary_mean_unit_cube(model)
  mean1 = boundary_mean_unit_cube(model, variant=1)
  coupling = high_flux_coupling_inf(
    pc_hi, X, E, boundary, model, n, 'neumann_a'
  )
  print(f'\nD={D} n={n} cells={cells} elements={len(E)} '
        f'exterior={len(boundary)} interfaces={ninterior}', flush=True)
  print(f'  physical boundary means: variant0={mean0:.12g}, '
        f'variant1={mean1:.12g}; '
        f'||M_beta^LH q_H||_inf={coupling:.3e}', flush=True)
  # The independent high-face coupling should be genuinely nonzero.
  assert coupling > args.min_high_coupling, (
    D, n, 'high-order coupling too small to validate', coupling
  )

  opts = dict(
    p_A=p_A, p_b=p_b, alpha=1.0, beta=0.0,
    assume_symmetric=True, tau_C=args.tau_C,
    leaf_operator_mode=HpsLeafOperatorMode.DENSE,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
  )

  def verify(solver, f, g, label, *, variant=0, zero_mean=False):
    result = solver.solve(f, g, check_residuals=True)
    assert_result(result, len(E), len(boundary), ninterior,
                  args.residual_tol)
    shift = (mean1 if variant else mean0) if zero_mean else 0.0
    rel, maxerr = error_to_exact(
      pc_eval, X, E, result.leaf_coeffs, model,
      variant=variant, shift=shift,
    )
    assert rel < args.solution_tol, (D, n, label, 'relative L2', rel)
    assert maxerr < args.solution_tol, (D, n, label, 'max error', maxerr)
    print(f'  {label:37s} relL2={rel:.3e} max={maxerr:.3e} '
          f'root={result.root_robin_residual_inf:.2e} '
          f'iface={result.interface_flux_residual_inf:.2e}', flush=True)

  def update(solver, alpha_modes, beta_modes):
    solver.update_variable_robin(
      alpha_modes, beta_modes, p_alpha=2, p_beta=2
    )

  def reject_bad_update(solver, label):
    try:
      update(solver, a_bad, b_bad)
    except RuntimeError as exc:
      print(f'  {label}: rejected ({exc})', flush=True)
    else:
      raise AssertionError('zero-alpha/zero-beta update was accepted')

  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A, b, None, keys, **opts
  ) as solver:
    verify(solver, f0, g_dir, 'initial Dirichlet')

    update(solver, a_neu, b_neu)
    verify(solver, f0, g_neu, 'Dirichlet -> variable Neumann', zero_mean=True)

    # Incompatible flux rejection must not poison the existing factorization.
    try:
      solver.solve(f0, g_incompatible, check_residuals=True)
    except RuntimeError as exc:
      print(f'  incompatible flux: rejected ({exc})', flush=True)
    else:
      raise AssertionError('incompatible pure Neumann data were accepted')
    verify(solver, f0, g_neu, 'Neumann after rejected flux', zero_mean=True)

    reject_bad_update(solver, 'invalid update in Neumann state')
    verify(solver, f0, g_neu, 'Neumann after rejected update', zero_mean=True)

    update(solver, a_rob, b_rob)
    verify(solver, f0, g_rob, 'Neumann -> variable Robin')

    reject_bad_update(solver, 'invalid update in Robin state')
    verify(solver, f0, g_rob, 'Robin after rejected update')

    # Changing BOTH boundary coefficients and the source exercises z_root,
    # including the high co-normal source response used in variable beta.
    update(solver, a_neu, b_neu2)
    verify(solver, f1, g_neu2, 'Robin -> different variable Neumann',
           variant=1, zero_mean=True)

    update(solver, a_rob2, b_rob2)
    verify(solver, f1, g_rob2, 'Neumann -> different variable Robin', variant=1)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--D', default='2,3',
                      help='Comma-separated dimensions, subset of 2,3')
  parser.add_argument('--n', default='5,6',
                      help='Comma-separated polynomial orders >= 5')
  parser.add_argument('--mesh-resolution', type=int, default=1)
  parser.add_argument('--solve-q-pad', type=int, default=2)
  parser.add_argument('--data-quad-extra', type=int, default=8)
  parser.add_argument('--tau-C', type=float, default=1.0)
  parser.add_argument('--omp-threads', type=int, default=4)
  parser.add_argument('--solution-tol', type=float, default=2e-8)
  parser.add_argument('--residual-tol', type=float, default=5e-8)
  parser.add_argument('--incompatible-offset', type=float, default=0.5)
  parser.add_argument('--min-high-coupling', type=float, default=1e-12)
  args = parser.parse_args()
  dims = [int(s) for s in args.D.split(',') if s.strip()]
  degrees = [int(s) for s in args.n.split(',') if s.strip()]
  if not dims or any(D not in (2, 3) for D in dims):
    parser.error('--D must be a comma-separated subset of 2,3')
  if not degrees or any(n < 5 for n in degrees):
    parser.error('--n must contain polynomial orders >= 5')
  if args.mesh_resolution < 1 or args.omp_threads < 1:
    parser.error('mesh resolution and OpenMP threads must be positive')
  set_openblas_threads(1)
  set_omp_threads(args.omp_threads)
  for D in dims:
    for n in degrees:
      run_one(D, n, args)
  print(f'\nPASS: high-order co-normal Robin/Neumann transitions '
        f'D={dims}, n={degrees}', flush=True)


if __name__ == '__main__':
  main()
