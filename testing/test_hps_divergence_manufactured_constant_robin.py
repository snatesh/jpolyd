"""Degree-5 homogeneous divergence-form HPS regression with constant Robin.

Usage (from the usual project Python environment):
  python testing/test_hps_divergence_manufactured_constant_robin.py --D 1,2,3
  python testing/test_hps_divergence_manufactured_constant_robin.py --D 4

This test is deliberately restricted to the persistent Dense/DenseQR path.
It uses physical co-normal flux and scalar (constant) alpha, beta. It does
not invoke the variable-Robin multiplication backend or high-order correction.
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
from jperms import face_sigma_array, face_vertices
from jprecomp import RefSimplexPrecomp, dimPi


class Manufactured:
  """Exact polynomial example from test_jdivergence.cpp: f == 0."""

  def __init__(self, D: int):
    self.D = D
    self.v = np.asarray([0.16 + 0.035 * j for j in range(D)])
    self.eta = np.asarray([0.12 + 0.025 * j for j in range(D)])
    self.w = np.asarray([0.35 - 0.055 * j for j in range(D)])
    self.t0 = 1.8
    self.epsilon = 0.07
    self.sigma = 0.4
    self.vv = float(self.v @ self.v)
    self.ew = float(self.eta @ self.w)

  def evaluate(self, x: np.ndarray):
    """Return u, grad(u), A, b, c and column-divergence(A) at physical x."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, self.D)
    t = self.t0 + x @ self.v
    s = x @ self.eta
    G = (
      np.eye(self.D)[None, :, :]
      + self.epsilon * (1.0 + s * s)[:, None, None]
      * np.outer(self.w, self.w)[None, :, :]
    )
    Gv = np.einsum("kij,j->ki", G, self.v)
    A = t[:, None, None] * G
    divA = (
      Gv + (2.0 * self.epsilon * t * s * self.ew)[:, None]
      * self.w[None, :]
    )
    b = (
      divA + 4.0 * Gv
      - (self.sigma * t * t / (5.0 * self.vv))[:, None]
      * self.v[None, :]
    )
    c = self.sigma * t
    u = t ** 5
    grad = 5.0 * t[:, None] ** 4 * self.v[None, :]
    return u, grad, A, b, c, divA

  def assert_pde(self, points: np.ndarray) -> None:
    """Independently check the physical divergence identity at test nodes."""
    u, grad, A, b, c, divA = self.evaluate(points)
    t = self.t0 + points @ self.v
    hessian = (20.0 * t ** 3)[:, None, None] * np.outer(
      self.v, self.v
    )[None, :, :]
    physical_residual = (
      -np.einsum("ki,ki->k", divA, grad)
      - np.einsum("kij,kij->k", A, hessian)
      + np.einsum("ki,ki->k", b, grad)
      + c * u
    )
    scale = max(1.0, float(np.max(np.abs(c * u))))
    assert float(np.max(np.abs(physical_residual))) / scale < 2e-13


def two_simplex_mesh(D: int):
  """Two well-shaped simplices sharing one interior face."""
  if D == 1:
    coords = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    simplices = np.array([[0, 1], [1, 2]], dtype=np.int32)
  else:
    coords = np.vstack((np.zeros((1, D)), np.eye(D), np.ones((1, D))))
    simplices = np.array([
      list(range(D + 1)),
      [D + 1, *range(1, D + 1)],
    ], dtype=np.int32)

  ids = np.arange(coords.shape[0], dtype=np.int32)
  merges = np.array([[0, 1]], dtype=np.int32)
  owners = {}
  for e, simplex in enumerate(simplices):
    for face_id in range(D + 1):
      key = tuple(sorted(np.delete(simplex, face_id).tolist()))
      owners.setdefault(key, []).append((e, face_id))
  assert sum(len(v) == 2 for v in owners.values()) == 1

  # Deliberately reorder faces relative to the usual sorted order.
  boundary = list(reversed(sorted(
    (key, owners_list[0])
    for key, owners_list in owners.items()
    if len(owners_list) == 1
  )))
  keys = np.asarray([key for key, _ in boundary], dtype=np.int32).reshape(-1, D)
  return ids, coords, simplices, merges, keys, boundary


def physical_volume_points(vertices: np.ndarray, Xhat: np.ndarray):
  B = (vertices[1:] - vertices[0]).T
  return vertices[0][None, :] + Xhat @ B.T


def face_points_normal_scale(
  D: int,
  vertices: np.ndarray,
  simplex: np.ndarray,
  face_id: int,
  Y: np.ndarray,
):
  """Canonical face quadrature, outward physical normal, area Jacobian."""
  Y = np.asarray(Y, dtype=np.float64).reshape(
    1 if D == 1 else -1, D - 1
  )
  bary_can = np.column_stack((1.0 - Y.sum(axis=1), Y))
  sigma = np.asarray(face_sigma_array(simplex, face_id), dtype=int)
  face_v = np.asarray(face_vertices(D, face_id), dtype=int)
  bary_vol = np.zeros((Y.shape[0], D + 1))
  for j in range(D):
    bary_vol[:, face_v[j]] = bary_can[:, sigma[j]]

  B = (vertices[1:] - vertices[0]).T
  points = vertices[0][None, :] + bary_vol[:, 1:] @ B.T
  n_ref = np.zeros(D)
  if face_id == 0:
    n_ref[:] = 1.0
  else:
    n_ref[face_id - 1] = -1.0
  normal = np.linalg.solve(B.T, n_ref)
  normal /= np.linalg.norm(normal)

  if D == 1:
    scale = 1.0
  else:
    canonical_vertices = np.empty((D, D))
    for j in range(D):
      canonical_vertices[:, sigma[j]] = vertices[face_v[j]]
    edges = canonical_vertices[:, 1:] - canonical_vertices[:, [0]]
    scale = math.sqrt(max(0.0, float(np.linalg.det(edges.T @ edges))))
  return points, normal, scale


def projected_volume_data(pc, coords, simplices, manufactured):
  """Independent base-kappa projection of original A, b, c and u*."""
  D = int(pc.D)
  MpA, MpB, MpC = (dimPi(D, p) for p in (3, 2, 1))
  Xhat, weights = pc.volume_quad()
  V = pc.volume_basis()
  nelem = len(simplices)
  A_data = np.empty((nelem, D, D, MpA))
  b_data = np.empty((nelem, D, MpB))
  c_data = np.empty((nelem, MpC))
  u_modal = np.empty((nelem, pc.M))

  for e, ids in enumerate(simplices):
    points = physical_volume_points(coords[ids], Xhat)
    manufactured.assert_pde(points)
    u, _grad, A, b, c, _divA = manufactured.evaluate(points)
    for r in range(D):
      for s in range(D):
        A_data[e, r, s] = V[:, :MpA].T @ (weights * A[:, r, s])
      b_data[e, r] = V[:, :MpB].T @ (weights * b[:, r])
    c_data[e] = V[:, :MpC].T @ (weights * c)
    u_modal[e] = V.T @ (weights * u)

  return (
    np.ascontiguousarray(A_data),
    np.ascontiguousarray(b_data),
    np.ascontiguousarray(c_data),
    np.ascontiguousarray(u_modal),
  )


def projected_constant_robin_data(
  pc, coords, simplices, boundary, manufactured, alpha, beta
):
  """Independent physical moments alpha*u + beta*n^T A grad(u)."""
  D = int(pc.D)
  Y, weights, V_face = pc.face_quad_basis()
  g = np.empty((len(boundary), pc.kf))
  max_flux = 0.0

  for row, (_key, (e, face_id)) in enumerate(boundary):
    ids = simplices[e]
    points, normal, scale = face_points_normal_scale(
      D, coords[ids], ids, face_id, Y
    )
    u, grad, A, _b, _c, _divA = manufactured.evaluate(points)
    flux = np.einsum("r,krs,ks->k", normal, A, grad)
    max_flux = max(max_flux, float(np.max(np.abs(flux))))
    g[row] = scale * (V_face.T @ (weights * (alpha * u + beta * flux)))

  assert max_flux > 1.0e-2  # Ensure beta actually tests co-normal flux.
  return np.ascontiguousarray(g), max_flux


def relative_weighted_l2(pc, coords, simplices, leaf_coeffs, manufactured):
  Xhat, weights = pc.volume_quad()
  V = pc.volume_basis()
  error2 = 0.0
  truth2 = 0.0
  for e, ids in enumerate(simplices):
    vertices = coords[ids]
    B = (vertices[1:] - vertices[0]).T
    jacobian = abs(float(np.linalg.det(B)))
    points = physical_volume_points(vertices, Xhat)
    u_true = manufactured.evaluate(points)[0]
    u_hps = V @ leaf_coeffs[e]
    error2 += jacobian * float(np.sum(weights * (u_hps - u_true) ** 2))
    truth2 += jacobian * float(np.sum(weights * u_true * u_true))
  return math.sqrt(error2 / max(truth2, 1.0e-300))


def test_dimension(D: int, n: int, error_tol: float, residual_tol: float):
  if n < 5:
    raise ValueError("the polynomial manufactured solution requires n >= 5")
  kappa = np.asarray([0.71 + 0.17 * j for j in range(D + 1)])
  pc = RefSimplexPrecomp(
    D, n, kappa,
    q_pad=4,
    q_vol=n + 4,
    q_face=1 if D == 1 else n + 4,
  )
  ids, coords, simplices, merges, keys, boundary = two_simplex_mesh(D)
  manufactured = Manufactured(D)
  A, b, c, u_exact = projected_volume_data(
    pc, coords, simplices, manufactured
  )
  alpha, beta = 1.25, 0.35
  g, max_flux = projected_constant_robin_data(
    pc, coords, simplices, boundary, manufactured, alpha, beta
  )
  f_physical = np.zeros((len(simplices), pc.M), dtype=np.float64)

  with HpsDivergenceSolver(
    pc,
    ids,
    coords,
    simplices,
    merges,
    A,
    b,
    c,
    keys,
    p_A=3,
    p_b=2,
    p_c=1,
    assume_symmetric=True,
    tau_C=1.0,
    alpha=alpha,
    beta=beta,
    leaf_operator_mode=HpsLeafOperatorMode.DENSE,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
  ) as solver:
    result = solver.solve(f_physical, g, check_residuals=True)

  modal_error = float(np.max(np.abs(result.leaf_coeffs - u_exact))) / max(
    1.0, float(np.max(np.abs(u_exact)))
  )
  l2_error = relative_weighted_l2(
    pc, coords, simplices, result.leaf_coeffs, manufactured
  )
  residuals = (
    result.root_robin_residual_inf,
    result.interface_flux_residual_inf,
    result.parent_consistency_residual_inf,
  )
  assert result.root_nb == len(boundary) * pc.kf, (D, result.root_nb)
  assert result.interface_nb == pc.kf, (D, result.interface_nb)
  assert np.all(np.isfinite(result.leaf_coeffs))
  assert np.all(np.isfinite(residuals)), (D, residuals)
  assert modal_error < error_tol, (D, "modal", modal_error)
  assert l2_error < error_tol, (D, "weighted L2", l2_error)
  assert max(residuals) < residual_tol, (D, "HPS residuals", residuals)

  print(
    f"D={D} n={n} PASS: "
    f"weighted L2={l2_error:.3e}, modal={modal_error:.3e}, "
    f"root={residuals[0]:.3e}, iface={residuals[1]:.3e}, "
    f"parent={residuals[2]:.3e}, max|q|={max_flux:.3e}"
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--D", default="1,2,3")
  parser.add_argument("--n", type=int, default=5)
  parser.add_argument("--error-tol", type=float, default=3.0e-8)
  parser.add_argument("--residual-tol", type=float, default=3.0e-8)
  args = parser.parse_args()
  for D in (int(v.strip()) for v in args.D.split(",") if v.strip()):
    if not 1 <= D <= 4:
      raise ValueError("focused HPS test supports D=1,...,4")
    test_dimension(D, args.n, args.error_tol, args.residual_tol)


if __name__ == "__main__":
  main()
