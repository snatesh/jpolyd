"""End-to-end divergence-form HPS conservation regression (variable Robin).

Compares ConservationMode None and ElementBalance for the original polynomial
manufactured problem in test_hps_divergence_variable_robin_polynomial.py.
The same variable A,b,c, nonzero physical source, variable alpha/beta, arbitrary
Jacobi kappa, and connected HPS mesh hierarchy are used in both modes.

Example:
  python testing/test_hps_divergence_conservation.py \
    --D 1,2,3 --min-n 5 --max-n 7 --mesh-resolution 1

Both modes' element-balance residuals are measured after the initial solve,
variable Robin update, and nonzero source/solution update. Only the constrained
mode must satisfy the balance tolerance. Exact polynomial
recovery is asserted for ElementBalance at n>=7, when the degree-seven source
and conormal flux are fully resolved. The unconstrained mode retains the
original n>=5 exactness checks. Lower-order conservative accuracy is recorded
but not constrained to match the unrestricted exact-polynomial solution.
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
  for row, (_key, (e, f)) in enumerate(boundary):
    P, normal, scale = face_points_normal_scale(
      D, coords[E[e]], E[e], f, Y
    )
    u, grad, _ = manufactured.u_grad_hess(P, solution_variant)
    A, _, _, _ = manufactured.coefficients(P)
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
    plt.figure(figsize=(8.5, 5.7))
    for D in sorted({int(r['D']) for r in rows}):
      drows = [r for r in rows if r['D'] == D]
      for mode, linestyle, marker in (
        ('none', '-', 'o'),
        ('element_balance', '--', 's'),
      ):
        sub = sorted((r for r in drows if r['mode'] == mode),
                     key=lambda r: r[abscissa])
        plt.semilogy([r[abscissa] for r in sub],
                     [max(r['relative_L2'], 1e-18) for r in sub],
                     linestyle=linestyle, marker=marker,
                     label=f'D={D}, {mode}')
      best = sorted((r for r in drows if r['mode'] == 'none'),
                    key=lambda r: r[abscissa])
      plt.semilogy([r[abscissa] for r in best],
                   [max(r['best_relative_L2'], 1e-18) for r in best],
                   linestyle=':', marker='.', alpha=.65,
                   label=f'D={D}, best projection')
    plt.grid(True, which='both', linewidth=.5)
    plt.xlabel(xlab)
    plt.ylabel('Relative weighted L2 error')
    plt.title('Divergence-form HPS: unconstrained vs element balance')
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f'divergence_conservation_{suffix}.png', dpi=200)
    plt.close()

  # Direct comparison of the same balance functional in both modes.
  plt.figure(figsize=(8.2, 5.5))
  for D in sorted({int(r['D']) for r in rows}):
    for mode, linestyle, marker in (('none', '-', 'o'),
                                    ('element_balance', '--', 's')):
      sub = sorted((r for r in rows if r['D'] == D and r['mode'] == mode),
                   key=lambda r: r['n'])
      plt.semilogy([r['n'] for r in sub],
                   [max(r['balance_residual_max'], 1e-18) for r in sub],
                   linestyle=linestyle, marker=marker,
                   label=f'D={D}, {mode}')
  plt.xlabel('Polynomial degree n')
  plt.ylabel('Maximum absolute element balance residual')
  plt.title('Discrete balance: unconstrained vs element balance')
  plt.grid(True, which='both', linewidth=.5)
  plt.legend(ncol=2, fontsize='small')
  plt.tight_layout()
  plt.savefig(output_dir / 'divergence_balance_comparison.png', dpi=200)
  plt.close()


def check_balance(solver, source, D, n, phase, args, enforce):
  """Read the most recently computed physical element-balance residuals."""
  balance = solver.balance_residuals()
  if balance.shape != (solver.nelem,):
    raise AssertionError((D, n, phase, 'balance shape', balance.shape))
  if not np.all(np.isfinite(balance)):
    raise AssertionError((D, n, phase, 'nonfinite balances', balance))
  max_abs = float(np.max(np.abs(balance)))
  l2 = float(np.linalg.norm(balance))
  signed = float(np.sum(balance))
  # The source array already contains the element Jacobian, as required by
  # the production HPS convention. This scale is only for the test tolerance.
  scale = max(1.0, float(np.max(np.abs(source))))
  if enforce and not max_abs < args.balance_tol * scale:
    raise AssertionError((D, n, phase, 'balance', max_abs, scale,
                          args.balance_tol))
  return max_abs, l2, signed


def run_dimension(D, args, rows):
  ids, X, E, merges, keys, boundary, ninterior, cells = cube_simplicial_mesh(
    D, args.mesh_resolution
  )
  print(f'\nD={D} grid={cells} vertices={len(ids)} elements={len(E)} '
        f'boundary_faces={len(boundary)} interfaces={ninterior} '
        f'merges={len(merges)}', flush=True)
  assert len(E) > 4
  model = Manufactured(D)
  kappa = np.asarray([0.71 + .17 * j for j in range(D + 1)], dtype=float)

  for n in range(args.min_n, args.max_n + 1):
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
    assert np.max(np.abs(f)) > 1e-12, (D, n, 'source is zero')

    solutions = {}
    for mode in ('none', 'element_balance'):
      with HpsDivergenceSolver(
        pc, ids, X, E, merges, A, b, c, keys,
        p_A=n, p_b=n, p_c=n,
        assume_symmetric=True, tau_C=args.tau_C,
        alpha=1.0, beta=0.0,
        leaf_operator_mode=HpsLeafOperatorMode.DENSE,
        leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
        conservation_mode=mode,
      ) as solver:
        solver.update_variable_robin(alpha, beta, p_alpha=n, p_beta=n)
        result = solver.solve(f, g, check_residuals=True)
        assert_result(result, len(E), len(boundary), ninterior,
                      args.residual_tol)
        error, best = relative_weighted_l2(pc_eval, X, E,
                                           result.leaf_coeffs, model)
        modal_err = float(np.max(np.abs(result.leaf_coeffs-u_exact))) / max(
          1.0, float(np.max(np.abs(u_exact)))
        )
        # Unconstrained regression reproduces original test's n>=5 criterion.
        # Constrained exactness is expected once source and co-normal data
        # (degree <=7) are fully resolved.
        exact_expected = n >= (5 if mode == 'none' else 7)
        if exact_expected:
          assert error < args.exact_tol, (D, n, mode, 'L2', error)
          assert modal_err < args.exact_tol, (D, n, mode, 'modal', modal_err)

        max_balance, l2_balance, signed_balance = check_balance(
          solver, f, D, n, 'initial', args,
          enforce=(mode == 'element_balance')
        )
        solutions[mode] = result.leaf_coeffs.copy()

        rows.append(dict(
          D=D, n=n, M=int(pc.M), mode=mode, elements=len(E),
          mesh_resolution=args.mesh_resolution,
          subdivisions='x'.join(map(str, cells)),
          relative_L2=error, best_relative_L2=best,
          modal_error=modal_err,
          root_residual=result.root_robin_residual_inf,
          interface_residual=result.interface_flux_residual_inf,
          parent_residual=result.parent_consistency_residual_inf,
          balance_residual_max=max_balance,
          balance_residual_l2=l2_balance,
          balance_residual_signed=signed_balance,
          max_physical_flux=max_flux,
        ))
        print(f'  n={n:2d} M={pc.M:3d} mode={mode:15s} '
              f'relL2={error:.3e} best={best:.3e} '
              f'modal={modal_err:.3e} '
              f'root={result.root_robin_residual_inf:.2e} '
              f'iface={result.interface_flux_residual_inf:.2e} '
              f'parent={result.parent_consistency_residual_inf:.2e} '
              f'balance={max_balance:.3e}', flush=True)

        if n in (5, 7):
          # Re-factor only the root variable-Robin matrix, no new HPS tree.
          a2, b2, g2, _ = project_boundary(
            pc_data, X, E, boundary, model, n, robin_variant=1
          )
          solver.update_variable_robin(a2, b2, p_alpha=n, p_beta=n)
          changed_robin = solver.solve(f, g2, check_residuals=True)
          assert_result(changed_robin, len(E), len(boundary), ninterior,
                        args.residual_tol)
          e1, _ = relative_weighted_l2(pc_eval, X, E,
                                       changed_robin.leaf_coeffs, model)
          if exact_expected:
            assert e1 < args.exact_tol, (D, n, mode, 'Robin update', e1)
          robin_balance, _, _ = check_balance(
            solver, f, D, n, 'Robin update', args,
            enforce=(mode == 'element_balance')
          )

          # Change source and exact solution, keeping operator and Robin data.
          _, _, _, f2, _ = project_volume(pc_data, X, E, model, n,
                                         variant=1,
                                         include_coefficients=False)
          _, _, g3, _ = project_boundary(pc_data, X, E, boundary, model, n,
                                        robin_variant=1,
                                        solution_variant=1)
          changed_source = solver.solve(f2, g3, check_residuals=True)
          assert_result(changed_source, len(E), len(boundary), ninterior,
                        args.residual_tol)
          e2, _ = relative_weighted_l2(pc_eval, X, E,
                                       changed_source.leaf_coeffs,
                                       model, variant=1)
          if exact_expected:
            assert e2 < args.exact_tol, (D, n, mode, 'source update', e2)
          source_balance, _, _ = check_balance(
            solver, f2, D, n, 'source update', args,
            enforce=(mode == 'element_balance')
          )
          print(f'    persistent updates [{mode}]: '
                f'robin L2={e1:.3e}, source L2={e2:.3e}, '
                f'balance Robin={robin_balance:.3e}, '
                f'balance source={source_balance:.3e}', flush=True)

    # Compare physical solution coefficients for identical first solve data.
    none = solutions['none']
    cons = solutions['element_balance']
    difference = float(np.linalg.norm(cons-none) /
                       max(1.0, float(np.linalg.norm(none))))
    print(f'  n={n:2d} relative constrained/unconstrained '
          f'coefficient difference={difference:.3e}', flush=True)
    for row in rows[-2:]:
      row['relative_mode_difference'] = difference


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--D', default='1,2,3',
                      help='Comma-separated dimensions (default 1,2,3)')
  parser.add_argument('--min-n', type=int, default=5)
  parser.add_argument('--max-n', type=int, default=7)
  parser.add_argument('--mesh-resolution', type=int, default=1,
                      help='Integer multiplier on D-dependent cube subdivisions')
  parser.add_argument('--solve-q-pad', type=int, default=2)
  parser.add_argument('--data-quad-extra', type=int, default=7)
  parser.add_argument('--tau-C', type=float, default=1.0)
  parser.add_argument('--omp-threads', type=int, default=4)
  parser.add_argument('--exact-tol', type=float, default=2e-9)
  parser.add_argument('--residual-tol', type=float, default=5e-8)
  parser.add_argument('--balance-tol', type=float, default=1e-8,
                      help='Max element balance <= balance_tol * max(1,max|f|)')
  parser.add_argument('--output-dir', type=Path,
                      default=Path('build/hps_divergence_conservation'))
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
  filename = args.output_dir / 'divergence_conservation_comparison.csv'
  with filename.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  plot_errors(rows, args.output_dir)
  print(f'\nPASS: conservative divergence-form variable-Robin HPS, '
        f'D={dims}, n={args.min_n}..{args.max_n}')
  print(f'CSV: {filename}')
  print(f'Plots: {args.output_dir}/divergence_conservation_vs_*.png')


if __name__ == '__main__':
  main()
