"""Regression: variable Robin with the existing tau-stabilized normal flux.

Place next to jhps.py and run from the usual project Python environment:
  python test_hps_elliptic_variable_robin.py --D 1,2,3
  python test_hps_elliptic_variable_robin.py --D 1,2,3,4

The test deliberately permutes its boundary-face input ordering.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from jhps import (
  HpsEllipticSolver,
  HpsLeafLeastSquaresSolver,
  HpsLeafOperatorMode,
)
from jperms import face_sigma_array, face_vertices
from jprecomp import RefSimplexPrecomp


def two_simplex_mesh(D: int):
  if D == 1:
    coords = np.array([[0.0], [1.0], [2.0]])
    simplices = np.array([[0, 1], [1, 2]], dtype=np.int32)
  else:
    coords = np.vstack((np.zeros((1, D)), np.eye(D), np.ones((1, D))))
    simplices = np.array([
      list(range(D + 1)),
      [D + 1, *range(1, D + 1)],
    ], dtype=np.int32)

  vertex_ids = np.arange(coords.shape[0], dtype=np.int32)
  merge_pairs = np.array([[0, 1]], dtype=np.int32)
  owners = {}
  for e, simplex in enumerate(simplices):
    for f in range(D + 1):
      key = tuple(sorted(np.delete(simplex, f).tolist()))
      owners.setdefault(key, []).append((e, f))
  boundary = sorted((key, value[0]) for key, value in owners.items() if len(value) == 1)
  # Different from canonical/root ordering: exercises the cached permutation.
  boundary = list(reversed(boundary))
  keys = np.asarray([row[0] for row in boundary], dtype=np.int32).reshape(-1, D)
  return vertex_ids, coords, simplices, merge_pairs, keys, boundary


def face_points_and_geometry(D, physical_vertices, simplex_ids, face_id, Y):
  """Canonical-face quadrature points, outward normal, physical area scaling."""
  Y = np.asarray(Y, dtype=float).reshape(
    1 if D == 1 else -1, D - 1
  )
  bary_can = np.column_stack((1.0 - Y.sum(axis=1), Y))
  sigma = np.asarray(face_sigma_array(simplex_ids, face_id), dtype=int)
  local_face_vertices = np.asarray(face_vertices(D, face_id), dtype=int)
  bary_vol = np.zeros((Y.shape[0], D + 1))
  for local in range(D):
    bary_vol[:, local_face_vertices[local]] = bary_can[:, sigma[local]]

  Xhat = bary_vol[:, 1:]
  B = physical_vertices[1:].T - physical_vertices[0][:, None]
  points = physical_vertices[0][None, :] + Xhat @ B.T

  # Reference face f is lambda_f=0; transform its outward covector by B^{-T}.
  n_ref = np.zeros(D)
  if face_id == 0:
    n_ref[:] = 1.0
  else:
    n_ref[face_id - 1] = -1.0
  outward = np.linalg.solve(B.T, n_ref)
  outward /= np.linalg.norm(outward)

  if D == 1:
    scale = 1.0
  else:
    face_vertices_canonical = np.empty((D, D))
    for local in range(D):
      face_vertices_canonical[:, sigma[local]] = physical_vertices[
        local_face_vertices[local]
      ]
    edges = face_vertices_canonical[:, 1:] - face_vertices_canonical[:, [0]]
    scale = math.sqrt(max(0.0, np.linalg.det(edges.T @ edges)))
  return points, outward, scale


def polynomial_u_and_grad(points):
  u = 1.0 + 0.17 * points.sum(axis=1) + 0.05 * np.sum(points**2, axis=1)
  grad = np.full_like(points, 0.17) + 0.1 * points
  if points.shape[1] > 1:
    u = u + 0.03 * points[:, 0] * points[:, 1]
    grad[:, 0] += 0.03 * points[:, 1]
    grad[:, 1] += 0.03 * points[:, 0]
  return u, grad


def alpha_function(points):
  return 1.6 + 0.11 * points[:, 0] + 0.07 * points.sum(axis=1)


def beta_function(points):
  return 0.35 + 0.07 * points[:, 0] + 0.02 * points.sum(axis=1)


def manufactured_boundary_data(pc, coords, simplices, boundary):
  D = int(pc.D)
  Y, w, V = pc.face_quad_basis()
  mp = math.comb(D, D - 1)  # dim Pi_1^{D-1} = D
  assert mp == D
  alpha = np.empty((len(boundary), mp))
  beta = np.empty_like(alpha)
  g = np.empty((len(boundary), pc.kf))

  for row, (_key, (e, local_f)) in enumerate(boundary):
    ids = simplices[e]
    points, normal, scale = face_points_and_geometry(
      D, coords[ids], ids, local_f, Y
    )
    u, grad = polynomial_u_and_grad(points)
    a_values = alpha_function(points)
    b_values = beta_function(points)
    physical_normal_derivative = grad @ normal
    g_values = a_values * u + b_values * physical_normal_derivative
    alpha[row] = V[:, :mp].T @ (w * a_values)
    beta[row] = V[:, :mp].T @ (w * b_values)
    # g is a physical face moment, hence the geometric face scale.
    g[row] = scale * (V.T @ (w * g_values))

  return np.ascontiguousarray(alpha), np.ascontiguousarray(beta), np.ascontiguousarray(g)


def expected_manufactured_data(pc, coords, simplices):
  D = int(pc.D)
  _Xres, w_res, Vres = pc.residual_quad_basis()
  Xvol, w_vol = pc.volume_quad()
  Vtrial = pc.volume_basis()
  exact_coeffs = np.empty((len(simplices), pc.M))
  source_coeffs = np.empty_like(exact_coeffs)
  const_source = -0.1 * D  # Laplacian of the manufactured quadratic is 0.1 D.

  for e, ids in enumerate(simplices):
    verts = coords[ids]
    B = (verts[1:] - verts[0]).T
    points = verts[0][None, :] + Xvol @ B.T
    u, _ = polynomial_u_and_grad(points)
    # Trial coefficients belong to kappa, source moments to kappa_res.
    exact_coeffs[e] = Vtrial.T @ (w_vol * u)
    source_coeffs[e] = abs(np.linalg.det(B)) * (
      Vres.T @ (w_res * const_source)
    )
  return np.ascontiguousarray(source_coeffs), exact_coeffs


def one_coefficient(pc):
  _Xhat, weights, Vres = pc.residual_quad_basis()
  one = float(weights @ Vres[:, 0])
  assert np.max(np.abs(Vres[:, 0] * one - 1.0)) < 1.0e-11
  return one


def constant_face_coefficient(pc):
  _Y, weights, Vface = pc.face_quad_basis()
  one = float(weights @ Vface[:, 0])
  assert np.max(np.abs(Vface[:, 0] * one - 1.0)) < 1.0e-11
  return one


def solve_and_check(solver, f, g):
  result = solver.solve(f, g, check_residuals=True)
  checks = np.array([
    result.root_robin_residual_inf,
    result.interface_flux_residual_inf,
    result.parent_consistency_residual_inf,
  ])
  assert np.all(np.isfinite(checks)), checks
  assert np.max(checks) < 2.0e-8, checks
  return result


def test_dimension(D, n=3):
  pc = RefSimplexPrecomp(
    D, n, np.asarray([0.71 + 0.17 * k for k in range(D + 1)]),
    q_pad=2, q_vol=n + 2, q_face=1 if D == 1 else n + 2,
  )
  vertex_ids, coords, simplices, merges, keys, boundary = two_simplex_mesh(D)
  nb = keys.shape[0]
  one_vol = one_coefficient(pc)
  one_face = constant_face_coefficient(pc)

  # The current primitive non-divergence solver: Delta u = -f.
  A = np.zeros((len(simplices), D, D, 1))
  for r in range(D):
    A[:, r, r, 0] = one_vol

  kwargs = dict(
    p2=0, p1=-1, p0=-1,
    assume_symmetric=True,
    tau_C=1.0,
    alpha=1.25, beta=0.65,
    leaf_operator_mode=HpsLeafOperatorMode.DENSE,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
  )

  rng = np.random.default_rng(12300 + D)
  random_f = np.ascontiguousarray(rng.normal(0, 0.03, (2, pc.M)))
  random_g = np.ascontiguousarray(rng.normal(0, 0.1, (nb, pc.kf)))
  constant_alpha = np.full((nb, 1), kwargs['alpha'] * one_face)
  constant_beta = np.full((nb, 1), kwargs['beta'] * one_face)

  with HpsEllipticSolver(
    pc, vertex_ids, coords, simplices, merges, A, None, None, keys,
    **kwargs,
  ) as scalar_solver, HpsEllipticSolver(
    pc, vertex_ids, coords, simplices, merges, A, None, None, keys,
    **kwargs,
  ) as variable_solver:
    scalar = solve_and_check(scalar_solver, random_f, random_g)

    # Test the new API and p=0 face Jacobi normalization.
    variable_solver.update_variable_robin(
      constant_alpha, constant_beta, p_alpha=0, p_beta=0
    )
    variable = solve_and_check(variable_solver, random_f, random_g)
    equivalence = np.max(np.abs(scalar.leaf_coeffs - variable.leaf_coeffs))
    scale = max(1.0, np.max(np.abs(scalar.leaf_coeffs)))
    assert equivalence / scale < 3.0e-9, (D, equivalence, scale)

    # Now vary both coefficients *along* each boundary face (D >= 2).
    a_face, b_face, manufactured_g = manufactured_boundary_data(
      pc, coords, simplices, boundary
    )
    manufactured_f, exact = expected_manufactured_data(pc, coords, simplices)
    variable_solver.update_variable_robin(
      a_face, b_face, p_alpha=1, p_beta=1
    )
    manufactured = solve_and_check(variable_solver, manufactured_f, manufactured_g)
    manufactured_error = np.max(np.abs(manufactured.leaf_coeffs - exact))
    exact_scale = max(1.0, np.max(np.abs(exact)))
    assert manufactured_error / exact_scale < 3.0e-8, (
      D, manufactured_error, exact_scale
    )

    # Update the same object back to constant coefficients: no leaf/tree rebuild.
    variable_solver.update_variable_robin(
      constant_alpha, constant_beta, p_alpha=0, p_beta=0
    )
    returned = solve_and_check(variable_solver, random_f, random_g)
    repeat_error = np.max(np.abs(scalar.leaf_coeffs - returned.leaf_coeffs))
    assert repeat_error / scale < 3.0e-9, (D, repeat_error)

  print(
    f'D={D} n={n} PASS: scalar/modal={equivalence:.3e}, '
    f'variable manufactured={manufactured_error:.3e}, '
    f'restored={repeat_error:.3e}'
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--D', default='1,2,3')
  parser.add_argument('--n', type=int, default=5)
  args = parser.parse_args()
  for D in (int(x.strip()) for x in args.D.split(',') if x.strip()):
    if not 1 <= D <= 4:
      raise ValueError('This focused test supports D=1,...,4')
    test_dimension(D, args.n)


if __name__ == '__main__':
  main()
