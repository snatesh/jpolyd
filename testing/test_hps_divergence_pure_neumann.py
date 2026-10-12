"""Pure-Neumann regression for the persistent divergence-form HPS solver.

Run from the usual jpolyd Python environment (jhps, jprecomp, jperms,
thread_control on PYTHONPATH, and rebuilt libjpolyd):

  python testing/test_hps_divergence_pure_neumann.py --D 1,2,3 --n 4

The domain is [0,1]^D, triangulated by connected Kuhn simplices. We verify:

  1. c=b=0, f=g=0: the zero-boundary-mean gauge returns u=0.
  2. c=b=0, u=sum(x_j**2): recover u minus its exact boundary mean.
  3. Incompatible unit outward flux (f=0, g=1): solve must fail, then a
     subsequent compatible solve on the same persistent handle must succeed.
  4. Constant nonzero drift, c=0: the left-nullspace compatibility and the
     constant-nullspace gauge still recover the manufactured quadratic.
  5. Positive constant reaction, c>0: pure-flux root is nonsingular and
     u=1 is recovered without imposing the zero-mean gauge.

IMPORTANT: HpsEllipticSolver.__init__ in jhps.py must allow alpha=0 when
beta!=0. This test exercises the SCALAR Robin path only; variable Robin
updates and nonzero configurable gauge RHS are separate follow-up work.
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
from jperms import face_sigma_array, face_vertices
from jprecomp import RefSimplexPrecomp
from thread_control import set_omp_threads, set_openblas_threads


BASE_SUBDIVISIONS = {1: (8,), 2: (2, 2), 3: (2, 1, 1)}


def cube_simplicial_mesh(D: int, resolution: int):
  """Connected, conforming Kuhn mesh with deliberately reversed face order."""
  cells = tuple(resolution * m for m in BASE_SUBDIVISIONS[D])
  grid = list(itertools.product(*(range(m + 1) for m in cells)))
  lookup = {v: i for i, v in enumerate(grid)}
  X = np.asarray(
    [[v[j] / cells[j] for j in range(D)] for v in grid], dtype=np.float64
  )
  simplices = []
  for cell in itertools.product(*(range(m) for m in cells)):
    for perm in itertools.permutations(range(D)):
      v = list(cell)
      simplex = [lookup[tuple(v)]]
      for j in perm:
        v[j] += 1
        simplex.append(lookup[tuple(v)])
      simplices.append(simplex)
  E = np.ascontiguousarray(simplices, dtype=np.int32)
  nelem = len(E)

  owners = {}
  for e, simplex in enumerate(E):
    for f in range(D + 1):
      key = tuple(sorted(np.delete(simplex, f).tolist()))
      owners.setdefault(key, []).append((e, f))
  assert all(len(o) <= 2 for o in owners.values())

  adjacency = [set() for _ in range(nelem)]
  for o in owners.values():
    if len(o) == 2:
      a, b = o[0][0], o[1][0]
      adjacency[a].add(b)
      adjacency[b].add(a)

  groups = {e: frozenset((e,)) for e in range(nelem)}
  active = list(range(nelem))
  merges = []
  while len(active) > 1:
    remaining = set(active)
    next_active = []
    for a in active:
      if a not in remaining:
        continue
      remaining.remove(a)
      neighbors = [
        b for b in active if b in remaining
        and any(adjacency[i] & groups[b] for i in groups[a])
      ]
      if not neighbors:
        next_active.append(a)
        continue
      b = min(neighbors, key=lambda j: (abs(len(groups[a]) - len(groups[j])), j))
      remaining.remove(b)
      parent = nelem + len(merges)
      merges.append((a, b))
      groups[parent] = groups[a] | groups[b]
      next_active.append(parent)
    if len(next_active) == len(active):
      raise AssertionError('Unable to construct a connected merge tree')
    active = next_active
  assert len(merges) == nelem - 1

  boundary = list(reversed(sorted(
    (key, o[0]) for key, o in owners.items() if len(o) == 1
  )))
  keys = np.asarray([key for key, _ in boundary], dtype=np.int32).reshape(-1, D)
  interior_count = sum(len(o) == 2 for o in owners.values())
  return (
    np.arange(len(X), dtype=np.int32), X, E,
    np.asarray(merges, dtype=np.int32), keys, boundary,
    interior_count, cells,
  )


def physical_volume_points(vertices, Xhat):
  B = (vertices[1:] - vertices[0]).T
  return vertices[0][None, :] + Xhat @ B.T, abs(float(np.linalg.det(B)))


def face_points_normal_scale(D, vertices, simplex, face_id, Y):
  """Physical points, outward unit normal and reference-face area Jacobian."""
  Y = np.asarray(Y, dtype=np.float64).reshape(1 if D == 1 else -1, D - 1)
  bary_can = np.column_stack((1.0 - Y.sum(axis=1), Y))
  sigma = np.asarray(face_sigma_array(simplex, face_id), dtype=int)
  face_v = np.asarray(face_vertices(D, face_id), dtype=int)
  bary_vol = np.zeros((len(Y), D + 1))
  for j in range(D):
    bary_vol[:, face_v[j]] = bary_can[:, sigma[j]]

  B = (vertices[1:] - vertices[0]).T
  P = vertices[0][None, :] + bary_vol[:, 1:] @ B.T
  nref = np.zeros(D)
  if face_id == 0:
    nref[:] = 1.0
  else:
    nref[face_id - 1] = -1.0
  normal = np.linalg.solve(B.T, nref)
  normal /= np.linalg.norm(normal)

  if D == 1:
    return P, normal, 1.0
  V = np.empty((D, D))
  for j in range(D):
    V[:, sigma[j]] = vertices[face_v[j]]
  edges = V[:, 1:] - V[:, [0]]
  scale = math.sqrt(max(0.0, float(np.linalg.det(edges.T @ edges))))
  return P, normal, scale


def assemble_data(pc_data, X, E, boundary, case, drift=None, reaction=0.0):
  """Project base-kappa A,b,c, residual-family f and physical face flux."""
  D = int(pc_data.D)
  nelem = len(E)
  kf = int(pc_data.kf)
  drift = None if drift is None else np.asarray(drift, dtype=np.float64)

  Xv, wv = pc_data.volume_quad()
  Vv = pc_data.volume_basis()
  Xr, wr, Vr = pc_data.residual_quad_basis()
  Y, wf, Vf = pc_data.face_quad_basis()

  # A=I, with p_A=0. Optional b and c are also degree-0 modal fields.
  one_coeff = float(Vv[:, 0] @ wv)
  A = np.zeros((nelem, D, D, 1), dtype=np.float64)
  for j in range(D):
    A[:, j, j, 0] = one_coeff
  b = None if drift is None else np.zeros((nelem, D, 1), dtype=np.float64)
  if b is not None:
    b[:, :, 0] = drift[None, :] * one_coeff
  c = None if reaction == 0 else np.full(
    (nelem, 1), float(reaction) * one_coeff, dtype=np.float64
  )

  f = np.empty((nelem, pc_data.M), dtype=np.float64)
  for e, ids in enumerate(E):
    P, detB = physical_volume_points(X[ids], Xr)
    if case == 'quadratic':
      physical_f = np.full(len(P), -2.0 * D)
      if drift is not None:
        physical_f += 2.0 * (P @ drift)
    elif case == 'reaction':
      physical_f = np.full(len(P), reaction)
    else:
      physical_f = np.zeros(len(P))
    # This is the physical f of -div(A grad u)+b.grad u+c*u=f.
    # The C++ divergence adapter negates these moments internally.
    f[e] = detB * (Vr.T @ (wr * physical_f))

  g = np.empty((len(boundary), kf), dtype=np.float64)
  for row, (_key, (e, face_id)) in enumerate(boundary):
    P, normal, scale = face_points_normal_scale(
      D, X[E[e]], E[e], face_id, Y
    )
    if case == 'quadratic':
      q = 2.0 * (P @ normal)
    elif case == 'incompatible':
      q = np.ones(len(P))  # Outward flux +1 on EVERY exterior face.
    else:
      q = np.zeros(len(P))
    # Face data are integrated against the canonical weighted face basis,
    # scaled by the physical face measure, as in the existing HPS tests.
    g[row] = scale * (Vf.T @ (wf * q))

  return (
    np.ascontiguousarray(A),
    None if b is None else np.ascontiguousarray(b),
    None if c is None else np.ascontiguousarray(c),
    np.ascontiguousarray(f),
    np.ascontiguousarray(g),
  )


def solution_error(pc_eval, X, E, coeffs, kind, boundary_mean=0.0):
  """Physical weighted-L2 and pointwise max errors at independent nodes."""
  Xhat, w = pc_eval.volume_quad()
  V = pc_eval.volume_basis()
  sum_error = 0.0
  sum_reference = 0.0
  max_error = 0.0
  for e, ids in enumerate(E):
    P, detB = physical_volume_points(X[ids], Xhat)
    target = (
      np.sum(P * P, axis=1) - boundary_mean
      if kind == 'quadratic'
      else np.full(len(P), 1.0 if kind == 'reaction' else 0.0)
    )
    error = V @ coeffs[e] - target
    sum_error += detB * float(np.sum(w * error * error))
    sum_reference += detB * float(np.sum(w * target * target))
    max_error = max(max_error, float(np.max(np.abs(error))))
  return math.sqrt(sum_error / max(sum_reference, 1.0)), max_error


def check_diagnostics(result, nelem, boundary_count, interface_count,
                      residual_tol):
  assert result.leaf_coeffs.shape == (nelem, result.M)
  assert result.root_nb == boundary_count * result.kf
  assert result.interface_nb == interface_count * result.kf
  diagnostic = np.asarray([
    result.root_robin_residual_inf,
    result.interface_flux_residual_inf,
    result.parent_consistency_residual_inf,
  ])
  assert np.all(np.isfinite(result.leaf_coeffs))
  assert np.all(np.isfinite(diagnostic)), diagnostic
  assert np.max(diagnostic) < residual_tol, diagnostic
  return float(np.max(diagnostic))


def run_dimension(D, args):
  ids, X, E, merges, keys, boundary, ni, cells = cube_simplicial_mesh(
    D, args.mesh_resolution
  )
  n = args.n
  kappa = np.asarray([0.68 + 0.11 * j for j in range(D + 1)])
  qsolve = n + args.solve_q_pad
  qdata = n + args.data_quad_extra
  pc = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.solve_q_pad,
    q_vol=qsolve, q_face=1 if D == 1 else qsolve,
  )
  pc_data = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.data_quad_extra,
    q_vol=qdata, q_face=1 if D == 1 else qdata,
  )
  pc_eval = RefSimplexPrecomp(
    D, n, kappa, q_pad=args.data_quad_extra + 2,
    q_vol=qdata + 2, q_face=1 if D == 1 else qdata + 2,
  )

  print(f'\nD={D}, n={n}, cells={cells}, elements={len(E)}, '
        f'boundary_faces={len(boundary)}, interfaces={ni}', flush=True)

  A, _, _, f_zero, g_zero = assemble_data(
    pc_data, X, E, boundary, 'zero'
  )
  _, _, _, f_quad, g_quad = assemble_data(
    pc_data, X, E, boundary, 'quadratic'
  )
  _, _, _, _, g_bad = assemble_data(
    pc_data, X, E, boundary, 'incompatible'
  )

  kwargs = dict(
    p_A=0, assume_symmetric=True,
    alpha=0.0, beta=1.0, tau_C=args.tau_C,
    leaf_operator_mode=HpsLeafOperatorMode.DENSE,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
  )

  def verify(result, expected_kind, label, mean=0.0):
    diagnostic = check_diagnostics(
      result, len(E), len(boundary), ni, args.residual_tol
    )
    relative, maximum = solution_error(
      pc_eval, X, E, result.leaf_coeffs, expected_kind, mean
    )
    assert relative < args.solution_tol, (D, label, 'weighted L2', relative)
    assert maximum < args.solution_tol, (D, label, 'max error', maximum)
    print(f'  {label:23s}: L2={relative:.3e}, max={maximum:.3e}, '
          f'residual={diagnostic:.2e}', flush=True)

  with HpsDivergenceSolver(
    pc, ids, X, E, merges, A, None, None, keys, **kwargs
  ) as solver:
    zero = solver.solve(f_zero, g_zero, check_residuals=True)
    verify(zero, 'zero', 'zero gauge / zero data')

    # Exact physical boundary mean of sum_j x_j^2 over the unit cube:
    #   integral_boundary u / area_boundary = (2D + 1) / 6.
    mean = (2.0 * D + 1.0) / 6.0
    quadratic = solver.solve(f_quad, g_quad, check_residuals=True)
    verify(quadratic, 'quadratic', 'quadratic, mean zero', mean)

    rejected = False
    try:
      solver.solve(f_zero, g_bad, check_residuals=True)
    except RuntimeError as exc:
      # This wrapper reports a C++ compatibility exception via a failure rc.
      assert 'solver_solve failed with rc=' in str(exc), str(exc)
      rejected = True
      print(f'  incompatible outward flux: rejected ({exc})', flush=True)
    assert rejected, 'Incompatible pure-Neumann data were incorrectly accepted'

    # A failed solve must not invalidate the stored tree/root factorization.
    again = solver.solve(f_quad, g_quad, check_residuals=True)
    verify(again, 'quadratic', 'compatible after failure', mean)

  if not args.skip_drift:
    drift = np.asarray([0.23 + 0.07 * j for j in range(D)])
    A, b, _, f_drift, g_drift = assemble_data(
      pc_data, X, E, boundary, 'quadratic', drift=drift
    )
    with HpsDivergenceSolver(
      pc, ids, X, E, merges, A, b, None, keys,
      p_b=0, **kwargs
    ) as solver:
      drift_result = solver.solve(f_drift, g_drift, check_residuals=True)
      verify(drift_result, 'quadratic', 'quadratic with drift', mean)

  if not args.skip_reaction:
    reaction = 0.6
    A, _, c, f_reaction, g_reaction = assemble_data(
      pc_data, X, E, boundary, 'reaction', reaction=reaction
    )
    with HpsDivergenceSolver(
      pc, ids, X, E, merges, A, None, c, keys,
      p_c=0, **kwargs
    ) as solver:
      result = solver.solve(f_reaction, g_reaction, check_residuals=True)
      # Unlike c=0, the Neumann problem is nonsingular: no gauge is applied.
      verify(result, 'reaction', 'positive reaction u=1')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--D', default='1,2,3',
                      help='Comma-separated dimension subset of 1,2,3')
  parser.add_argument('--n', type=int, default=4)
  parser.add_argument('--mesh-resolution', type=int, default=1)
  parser.add_argument('--solve-q-pad', type=int, default=2)
  parser.add_argument('--data-quad-extra', type=int, default=7)
  parser.add_argument('--tau-C', type=float, default=1.0)
  parser.add_argument('--omp-threads', type=int, default=4)
  parser.add_argument('--solution-tol', type=float, default=2e-8)
  parser.add_argument('--residual-tol', type=float, default=5e-8)
  parser.add_argument('--skip-drift', action='store_true')
  parser.add_argument('--skip-reaction', action='store_true')
  args = parser.parse_args()
  dims = [int(x) for x in args.D.split(',') if x.strip()]
  if not dims or any(D not in BASE_SUBDIVISIONS for D in dims):
    parser.error('D must be a comma-separated subset of 1,2,3')
  if args.n < 2 or args.mesh_resolution < 1 or args.solve_q_pad < 0:
    parser.error('require n>=2, mesh-resolution>=1 and solve-q-pad>=0')
  if args.data_quad_extra < 3 or args.omp_threads < 1:
    parser.error('require data-quad-extra>=3 and omp-threads>=1')
  set_openblas_threads(1)
  set_omp_threads(args.omp_threads)
  for D in dims:
    run_dimension(D, args)
  print(f'\nPASS: pure-Neumann persistent divergence HPS D={dims}, n={args.n}',
        flush=True)


if __name__ == '__main__':
  main()
