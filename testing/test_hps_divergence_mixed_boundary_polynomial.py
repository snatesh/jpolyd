"""Dense divergence-form HPS: polynomial recovery with mixed boundary data.

Run from the project environment (jhps, jprecomp and jperms on PYTHONPATH):

  python testing/test_hps_divergence_mixed_boundary_polynomial.py \
    --D 1,2,3 --min-n 3 --max-n 7 --mesh-resolution 1

The unit D-cube is triangulated with Kuhn/Freudenthal simplices. The integer
mesh-resolution factor multiplies the baseline grid subdivisions:
  D=1: (8,), D=2: (2,2), D=3: (2,1,1).
For resolution=1 this gives 8/8/12 elements respectively. The grid and merge
structure are fixed as n varies within each dimension.

Input A,b,c and alpha,beta are all *stored* to degree n, while their physical
polynomial degrees stay fixed (A=3, b=2, c=2, alpha=2, beta=2).
The fixed manufactured solution has degree 5 and the physical source is nonzero.

Boundary conditions are assigned geometrically on the unit D-cube:
  x[0] == 0: pure Neumann (alpha=0, beta=1)
  x[0] == 1: Dirichlet    (alpha=1, beta=0)
  other faces: spatially variable Robin (alpha>0, beta>0)
For D=1 there are only Neumann and Dirichlet endpoints. The exact g is
projected from alpha*u + beta*n^T A grad(u) for all faces. Persistent Robin
updates retain the same boundary-condition types.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from jhps import HpsDivergenceSolver, HpsLeafOperatorMode, HpsLeafLeastSquaresSolver
from jperms import face_sigma_array, face_vertices
from jprecomp import RefSimplexPrecomp, dimPi
from thread_control import set_omp_threads, set_openblas_threads


BASE_SUBDIVISIONS = {1: (8,), 2: (2, 2), 3: (2, 1, 1)}


def cube_simplicial_mesh(D: int, resolution: int):
  """Conforming oriented Kuhn triangulation of the subdivided unit cube."""
  if D not in BASE_SUBDIVISIONS or resolution < 1:
    raise ValueError('D must be 1..3 and mesh resolution must be >= 1')
  cells = tuple(resolution * m for m in BASE_SUBDIVISIONS[D])
  grid = list(itertools.product(*(range(m + 1) for m in cells)))
  lookup = {v: i for i, v in enumerate(grid)}
  X = np.asarray([[v[j] / cells[j] for j in range(D)] for v in grid], dtype=np.float64)
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
  nelem = E.shape[0]
  face_owners = {}
  for e, simplex in enumerate(E):
    for f in range(D + 1):
      key = tuple(sorted(np.delete(simplex, f).tolist()))
      face_owners.setdefault(key, []).append((e, f))
  if any(len(v) > 2 for v in face_owners.values()):
    raise AssertionError('Nonmanifold simplex mesh')

  adj = [set() for _ in range(nelem)]
  for owners in face_owners.values():
    if len(owners) == 2:
      a, b = owners[0][0], owners[1][0]
      adj[a].add(b)
      adj[b].add(a)
  # Each merge combines adjacent, connected partitions. No PyMetis dependency.
  groups = {i: frozenset((i,)) for i in range(nelem)}
  active = list(range(nelem))
  pairs = []
  while len(active) > 1:
    remaining = set(active)
    next_active = []
    for a in active:
      if a not in remaining:
        continue
      remaining.remove(a)
      neighbors = [
        b for b in active if b in remaining and
        any(bool(adj[i] & groups[b]) for i in groups[a])
      ]
      if not neighbors:
        next_active.append(a)
        continue
      b = min(neighbors, key=lambda j: (abs(len(groups[a]) - len(groups[j])), j))
      remaining.remove(b)
      parent = nelem + len(pairs)
      pairs.append((a, b))
      groups[parent] = groups[a] | groups[b]
      next_active.append(parent)
    if len(next_active) >= len(active):
      raise AssertionError('Could not construct a connected HPS merge tree')
    active = next_active
  assert len(pairs) == nelem - 1

  # Reverse input face order intentionally to test the cached permutation.
  boundary = list(reversed(sorted(
    (key, owners[0]) for key, owners in face_owners.items() if len(owners) == 1
  )))
  keys = np.asarray([key for key, _ in boundary], dtype=np.int32).reshape(-1, D)
  ninterior = sum(len(owners) == 2 for owners in face_owners.values())
  return (
    np.arange(len(X), dtype=np.int32), X, E,
    np.asarray(pairs, dtype=np.int32), keys, boundary, ninterior, cells
  )


def volume_points(vertices, Xhat):
  B = (vertices[1:] - vertices[0]).T
  return vertices[0][None, :] + Xhat @ B.T, abs(float(np.linalg.det(B)))


def face_points_normal_scale(D, vertices, ids, face_id, Y):
  Y = np.asarray(Y, dtype=np.float64).reshape(1 if D == 1 else -1, D - 1)
  bary_can = np.column_stack((1.0 - np.sum(Y, axis=1), Y))
  sigma = np.asarray(face_sigma_array(ids, face_id), dtype=int)
  local_face_vertices = np.asarray(face_vertices(D, face_id), dtype=int)
  bary_vol = np.zeros((len(Y), D + 1), dtype=np.float64)
  for i in range(D):
    bary_vol[:, local_face_vertices[i]] = bary_can[:, sigma[i]]
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
  for i in range(D):
    V[:, sigma[i]] = vertices[local_face_vertices[i]]
  edges = V[:, 1:] - V[:, [0]]
  scale = math.sqrt(max(0., float(np.linalg.det(edges.T @ edges))))
  return P, normal, scale


class Manufactured:
  """One fixed degree-five polynomial and polynomial PDE coefficients."""

  def __init__(self, D):
    self.D = D
    self.v = np.asarray([0.18 + 0.055 * j for j in range(D)])
    self.eta = np.asarray([0.13 + 0.03 * j for j in range(D)])
    self.w = np.asarray([0.42 - 0.065 * j for j in range(D)])
    self.epsilon = 0.09
    self.terms = []
    for j in range(D):
      p = [0] * D
      p[j] = 5
      self.terms.append((0.014 * (j + 1), tuple(p)))
    if D >= 2:
      self.terms.append((0.026, (2, 3) + (0,) * (D - 2)))
    if D == 3:
      self.terms.append((-0.019, (1, 2, 2)))

  def u_grad_hess(self, X, variant=0):
    N, D = X.shape
    t = 1.20 + X @ self.v
    u = t ** 5
    grad = 5.0 * t[:, None] ** 4 * self.v[None, :]
    hess = 20.0 * t[:, None, None] ** 3 * np.outer(self.v, self.v)[None, :, :]
    for coef, power in self.terms:
      def term(deriv):
        powers = list(power)
        prefactor = coef
        for d in deriv:
          if powers[d] == 0:
            return np.zeros(N)
          prefactor *= powers[d]
          powers[d] -= 1
        out = np.full(N, prefactor)
        for d, exponent in enumerate(powers):
          if exponent:
            out *= X[:, d] ** exponent
        return out
      u += term(())
      for a in range(D):
        grad[:, a] += term((a,))
        for b in range(D):
          hess[:, a, b] += term((a, b))
    if variant:
      # A different degree<=5 solution; PDE coefficients remain fixed.
      factor = 0.065
      u += factor * np.sum(X ** 3, axis=1)
      grad += 3.0 * factor * X ** 2
      for d in range(D):
        hess[:, d, d] += 6.0 * factor * X[:, d]
    return u, grad, hess

  def coefficients(self, X):
    D = self.D
    t = 1.20 + X @ self.v
    s = X @ self.eta
    G = np.eye(D)[None, :, :] + (
      self.epsilon * (1.0 + s ** 2)[:, None, None]
      * np.outer(self.w, self.w)[None, :, :]
    )
    A = t[:, None, None] * G
    divA = G @ self.v + (
      2.0 * self.epsilon * t * s * (self.eta @ self.w)
    )[:, None] * self.w[None, :]
    b = (
      0.07 * (1.0 + 0.35 * np.sum(X * X, axis=1))[:, None]
      * np.asarray([1.0 + 0.15 * j for j in range(D)])[None, :]
      + 0.045 * s[:, None] ** 2 * self.v[None, :]
    )
    c = 0.35 + 0.08 * np.sum(X ** 2, axis=1)
    return A, divA, b, c

  def robin(self, X, variant=0):
    ss = np.sum(X, axis=1)
    ss2 = np.sum(X ** 2, axis=1)
    alpha = 1.25 + 0.10 * ss + 0.055 * ss2
    beta = 0.48 + 0.07 * X[:, 0] + 0.075 * ss2
    if variant:
      alpha = alpha + 0.19 + 0.035 * X[:, 0] ** 2
      beta = 0.82 * beta + 0.032 * ss
    return alpha, beta

  def physical_source(self, X, variant=0):
    u, grad, hess = self.u_grad_hess(X, variant)
    A, divA, b, c = self.coefficients(X)
    return (
      -np.einsum('ki,ki->k', divA, grad)
      -np.einsum('kij,kij->k', A, hess)
      +np.einsum('ki,ki->k', b, grad) + c * u
    )


def project_volume(pc_data, coords, E, manufactured, n, variant=0,
                   include_coefficients=True):
  D = manufactured.D
  Mp = dimPi(D, n)
  X, w = pc_data.volume_quad()
  V = pc_data.volume_basis()[:, :Mp]
  Xr, wr, Vr = pc_data.residual_quad_basis()
  Vr = Vr[:, :Mp]
  ne = len(E)
  Aout = np.empty((ne, D, D, Mp)) if include_coefficients else None
  bout = np.empty((ne, D, Mp)) if include_coefficients else None
  cout = np.empty((ne, Mp)) if include_coefficients else None
  uout = np.empty((ne, Mp))
  fout = np.empty((ne, Mp))
  for e, ids in enumerate(E):
    xv, detB = volume_points(coords[ids], X)
    xr, _ = volume_points(coords[ids], Xr)
    u, _, _ = manufactured.u_grad_hess(xv, variant)
    uout[e] = V.T @ (w * u)
    fout[e] = detB * (Vr.T @ (wr * manufactured.physical_source(xr, variant)))
    if include_coefficients:
      A, _, b, c = manufactured.coefficients(xv)
      for r in range(D):
        for s in range(D):
          Aout[e, r, s] = V.T @ (w * A[:, r, s])
        bout[e, r] = V.T @ (w * b[:, r])
      cout[e] = V.T @ (w * c)
  def contig(a):
    return None if a is None else np.ascontiguousarray(a)
  return tuple(map(contig, (Aout, bout, cout, fout, uout)))


def boundary_condition_type(coords, key):
  """Classify a cube boundary face from its vertices, not quadrature nodes."""
  face_vertices = coords[np.asarray(key, dtype=np.intp)]
  if np.all(np.abs(face_vertices[:, 0]) < 1e-12):
    return 'neumann'
  if np.all(np.abs(face_vertices[:, 0] - 1.0) < 1e-12):
    return 'dirichlet'
  return 'robin'


def project_boundary(pc_data, coords, E, boundary, manufactured, n,
                     robin_variant=0, solution_variant=0):
  D = manufactured.D
  Y, w, V = pc_data.face_quad_basis()
  k_beta = dimPi(D - 1, n)
  Vc = V[:, :k_beta]
  alpha_coeffs = np.empty((len(boundary), k_beta))
  beta_coeffs = np.empty_like(alpha_coeffs)
  g = np.empty((len(boundary), pc_data.kf))
  flux_max = 0.0
  for row, (key, (e, f)) in enumerate(boundary):
    P, normal, scale = face_points_normal_scale(
      D, coords[E[e]], E[e], f, Y
    )
    u, grad, _ = manufactured.u_grad_hess(P, solution_variant)
    A, _, _, _ = manufactured.coefficients(P)
    kind = boundary_condition_type(coords, key)
    if kind == 'neumann':
      alpha = np.zeros(len(P), dtype=np.float64)
      beta = np.ones(len(P), dtype=np.float64)
    elif kind == 'dirichlet':
      alpha = np.ones(len(P), dtype=np.float64)
      beta = np.zeros(len(P), dtype=np.float64)
    else:
      alpha, beta = manufactured.robin(P, robin_variant)
    flux = np.einsum('i,kij,kj->k', normal, A, grad)
    alpha_coeffs[row] = Vc.T @ (w * alpha)
    beta_coeffs[row] = Vc.T @ (w * beta)
    g[row] = scale * (V.T @ (w * (alpha * u + beta * flux)))
    flux_max = max(flux_max, float(np.max(np.abs(flux))))
  return (np.ascontiguousarray(alpha_coeffs),
          np.ascontiguousarray(beta_coeffs),
          np.ascontiguousarray(g), flux_max)


def relative_weighted_l2(pc_eval, coords, E, coefficients, manufactured, variant=0):
  X, w = pc_eval.volume_quad()
  V = pc_eval.volume_basis()
  err = exact = best = 0.0
  for e, ids in enumerate(E):
    P, det = volume_points(coords[ids], X)
    u, _, _ = manufactured.u_grad_hess(P, variant)
    uh = V @ coefficients[e]
    diff = uh - u
    err += det * float(np.sum(w * diff * diff))
    exact += det * float(np.sum(w * u * u))
    projection = V.T @ (w * u)
    best += det * float(np.sum(w * (V @ projection - u) ** 2))
  return math.sqrt(err / max(exact, 1e-300)), math.sqrt(best / max(exact, 1e-300))


def assert_result(result, nelem, boundary_count, interior_count, tol):
  assert result.root_nb == boundary_count * result.kf, (
    'root faces', result.root_nb, boundary_count, result.kf
  )
  assert result.interface_nb == interior_count * result.kf, (
    'interior faces', result.interface_nb, interior_count, result.kf
  )
  diagnostics = np.array((
    result.root_robin_residual_inf,
    result.interface_flux_residual_inf,
    result.parent_consistency_residual_inf,
  ))
  assert np.all(np.isfinite(diagnostics)), diagnostics
  assert np.all(np.isfinite(result.leaf_coeffs))
  assert np.max(diagnostics) < tol, diagnostics
  assert result.leaf_coeffs.shape[0] == nelem


def plot_errors(rows, output_dir):
  for abscissa, xlab, suffix in (
    ('n', 'Polynomial degree n', 'vs_n'),
    ('M', 'M = dim Pi_n^D', 'vs_dimPi'),
  ):
    plt.figure(figsize=(8.2, 5.5))
    for D in sorted({int(r['D']) for r in rows}):
      sub = sorted((r for r in rows if r['D'] == D), key=lambda r: r[abscissa])
      xx = [r[abscissa] for r in sub]
      yy = [max(r['relative_L2'], 1e-18) for r in sub]
      bb = [max(r['best_relative_L2'], 1e-18) for r in sub]
      line, = plt.semilogy(xx, yy, marker='o', label=f'HPS D={D}')
      plt.semilogy(xx, bb, marker='s', linestyle='--', color=line.get_color(),
                   alpha=0.7, label=f'Best projection D={D}')
    plt.grid(True, which='both', linewidth=.5)
    plt.xlabel(xlab)
    plt.ylabel('Relative weighted L2 error')
    plt.title('Dense divergence-form HPS, mixed BC, degree-5 solution')
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(output_dir / f'mixed_boundary_polynomial_{suffix}.png', dpi=200)
    plt.close()


def run_dimension(D, args, rows):
  ids, X, E, merges, keys, boundary, ninterior, cells = cube_simplicial_mesh(
    D, args.mesh_resolution
  )
  print(f'\nD={D} grid={cells} vertices={len(ids)} elements={len(E)} '
        f'boundary_faces={len(boundary)} interfaces={ninterior} '
        f'merges={len(merges)}', flush=True)
  assert len(E) > 4
  bc_counts = {name: sum(boundary_condition_type(X, key) == name
                         for key, _ in boundary)
               for name in ('neumann', 'dirichlet', 'robin')}
  assert bc_counts['neumann'] > 0 and bc_counts['dirichlet'] > 0
  if D >= 2:
    assert bc_counts['robin'] > 0
  print('  boundary types: ' + ', '.join(f'{name}={count}'
        for name, count in bc_counts.items()), flush=True)
  model = Manufactured(D)
  kappa = np.asarray([0.71 + .17 * j for j in range(D + 1)], dtype=float)
  for n in range(args.min_n, args.max_n + 1):
    # Accurate projection of the degree<=9 boundary data and degree<=7 source.
    q_data = n + args.data_quad_extra
    q_face = 1 if D == 1 else q_data
    pc = RefSimplexPrecomp(D, n, kappa, q_pad=args.solve_q_pad,
                           q_vol=n + args.solve_q_pad,
                           q_face=1 if D == 1 else n + args.solve_q_pad)
    pc_data = RefSimplexPrecomp(D, n, kappa, q_pad=args.data_quad_extra,
                                q_vol=q_data, q_face=q_face)
    pc_eval = RefSimplexPrecomp(D, n, kappa,
                                q_pad=args.data_quad_extra + 2,
                                q_vol=q_data + 2,
                                q_face=1 if D == 1 else q_data + 2)
    A, b, c, f, u_exact = project_volume(pc_data, X, E, model, n)
    alpha, beta, g, max_flux = project_boundary(pc_data, X, E, boundary,
                                                model, n)
    with HpsDivergenceSolver(
      pc, ids, X, E, merges, A, b, c, keys,
      p_A=n, p_b=n, p_c=n,
      assume_symmetric=True, tau_C=args.tau_C,
      alpha=1.0, beta=0.0,
      leaf_operator_mode=HpsLeafOperatorMode.DENSE,
      leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
    ) as solver:
      solver.update_variable_robin(alpha, beta, p_alpha=n, p_beta=n)
      result = solver.solve(f, g, check_residuals=True)
      assert_result(result, len(E), len(boundary), ninterior,
                    args.residual_tol)
      error, best = relative_weighted_l2(pc_eval, X, E,
                                         result.leaf_coeffs, model)
      modal_err = float(np.max(np.abs(result.leaf_coeffs - u_exact))) / max(
        1.0, float(np.max(np.abs(u_exact)))
      )
      if n >= 5:
        assert error < args.exact_tol, (D, n, 'L2', error)
        assert modal_err < args.exact_tol, (D, n, 'modal', modal_err)
      rows.append(dict(D=D, n=n, M=int(pc.M), elements=len(E),
                       mesh_resolution=args.mesh_resolution,
                       subdivisions='x'.join(map(str, cells)),
                       relative_L2=error, best_relative_L2=best,
                       modal_error=modal_err,
                       root_residual=result.root_robin_residual_inf,
                       interface_residual=result.interface_flux_residual_inf,
                       parent_residual=result.parent_consistency_residual_inf,
                       max_physical_flux=max_flux,
                       neumann_faces=bc_counts['neumann'],
                       dirichlet_faces=bc_counts['dirichlet'],
                       robin_faces=bc_counts['robin']))
      print(f'  n={n:2d} M={pc.M:3d} relL2={error:.3e} '
            f'best={best:.3e} modal={modal_err:.3e} '
            f'root={result.root_robin_residual_inf:.2e} '
            f'iface={result.interface_flux_residual_inf:.2e} '
            f'parent={result.parent_consistency_residual_inf:.2e}', flush=True)
      if n == 5:
        # Re-factor just the Robin root matrix, then refresh source data for a
        # second global polynomial, without rebuilding any leaf/HPS matrices.
        a2, b2, g2, _ = project_boundary(
          pc_data, X, E, boundary, model, n, robin_variant=1
        )
        solver.update_variable_robin(a2, b2, p_alpha=n, p_beta=n)
        changed_robin = solver.solve(f, g2)
        assert_result(changed_robin, len(E), len(boundary), ninterior,
                      args.residual_tol)
        e1, _ = relative_weighted_l2(pc_eval, X, E,
                                     changed_robin.leaf_coeffs, model)
        assert e1 < args.exact_tol, (D, 'Robin update', e1)

        _, _, _, f2, _ = project_volume(pc_data, X, E, model, n,
                                       variant=1, include_coefficients=False)
        _, _, g3, _ = project_boundary(pc_data, X, E, boundary, model, n,
                                      robin_variant=1, solution_variant=1)
        changed_source = solver.solve(f2, g3)
        assert_result(changed_source, len(E), len(boundary), ninterior,
                      args.residual_tol)
        e2, _ = relative_weighted_l2(pc_eval, X, E,
                                     changed_source.leaf_coeffs, model, variant=1)
        assert e2 < args.exact_tol, (D, 'Source update', e2)
        print(f'    persistent updates: robin={e1:.3e}, '
              f'nonzero-source={e2:.3e}', flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--D', default='1,2,3',
                      help='Comma-separated dimensions (default 1,2,3)')
  parser.add_argument('--min-n', type=int, default=3)
  parser.add_argument('--max-n', type=int, default=7)
  parser.add_argument('--mesh-resolution', type=int, default=1,
                      help='Integer multiplier on D-dependent cube subdivisions '
                           '(baseline 8, 2x2, 2x1x1; default 1)')
  parser.add_argument('--solve-q-pad', type=int, default=2)
  parser.add_argument('--data-quad-extra', type=int, default=7)
  parser.add_argument('--tau-C', type=float, default=1.0)
  parser.add_argument('--omp-threads', type=int, default=4)
  parser.add_argument('--exact-tol', type=float, default=2e-9)
  parser.add_argument('--residual-tol', type=float, default=5e-8)
  parser.add_argument('--output-dir', type=Path,
                      default=Path('build/hps_divergence_mixed_boundary_poly'))
  args = parser.parse_args()
  if args.min_n < 2 or args.max_n < args.min_n or args.mesh_resolution < 1:
    parser.error('require 2 <= min-n <= max-n and mesh-resolution >= 1')
  if args.omp_threads < 1:
    parser.error('omp-threads must be >= 1')
  if args.data_quad_extra < 5:
    parser.error('data-quad-extra must be >= 5 for exact polynomial projections')
  dims = [int(v) for v in args.D.split(',') if v.strip()]
  if not dims or any(D not in (1, 2, 3) for D in dims):
    parser.error('D must be a comma-separated subset of 1,2,3')
  set_openblas_threads(1)
  set_omp_threads(args.omp_threads)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for D in dims:
    run_dimension(D, args, rows)
  filename = args.output_dir / 'mixed_boundary_polynomial_convergence.csv'
  with filename.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  plot_errors(rows, args.output_dir)
  print(f'\nPASS: polynomial mixed-boundary Dense divergence-form tests, '
        f'D={dims}, n={args.min_n}..{args.max_n}')
  print(f'CSV: {filename}')
  print(f'Plots: {args.output_dir}/mixed_boundary_polynomial_vs_*.png')


if __name__ == '__main__':
  main()
