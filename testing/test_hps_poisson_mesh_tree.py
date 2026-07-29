from __future__ import annotations

import argparse
import importlib.metadata
import math
from pathlib import Path
from typing import Callable

import numpy as np

from jhps import HpsPoissonResult, run_poisson_mesh_tree_solve
from hps_mesh_tree_driver import (
  DEFAULT_QHULL_OPTIONS,
  build_merge_tree,
  export_case,
  generate_mesh,
  validate_merge_tree,
  visualize_case,
)
from jperms import face_sigma_array, face_vertices
from jprecomp import RefSimplexPrecomp


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "python" else _THIS.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "build" / "hps_poisson_mesh_tree_test"

DEFAULT_INTERIOR_POINT_COUNTS = {
  1: 4,
  2: 1,
  3: 0,
  4: 0,
  5: 0,
}

ScalarFun = Callable[[np.ndarray], np.ndarray]
GradFun = Callable[[np.ndarray], np.ndarray]


def parse_D_list(text: str) -> list[int]:
  if text.strip().lower() == "all":
    return [1, 2, 3, 4, 5]
  out = [int(part.strip()) for part in text.split(",") if part.strip()]
  if not out or any(D < 1 or D > 5 for D in out):
    raise ValueError("--D must list dimensions in 1..5")
  return out


def manufactured_polynomial(D: int) -> tuple[ScalarFun, ScalarFun, GradFun]:
  """Return a global cubic u, f=-Delta u, and grad u."""
  a = np.array([0.13 * (i + 1) for i in range(D)], dtype=np.float64)
  b = np.array([0.07 + 0.025 * i for i in range(D)], dtype=np.float64)
  c = np.array([0.015 * ((-1.0) ** i) * (i + 1) for i in range(D)], dtype=np.float64)
  cross = np.zeros((D, D), dtype=np.float64)
  for i in range(D):
    for j in range(i + 1, D):
      cross[i, j] = cross[j, i] = 0.02 * (i + 1) / (j + 2)

  def u_fun(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    out = np.ones(P.shape[0], dtype=np.float64)
    out += P @ a
    out += (P * P) @ b
    out += (P * P * P) @ c
    for i in range(D):
      for j in range(i + 1, D):
        out += cross[i, j] * P[:, i] * P[:, j]
    return out

  def f_fun(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    lap = np.full(P.shape[0], 2.0 * float(np.sum(b)), dtype=np.float64)
    lap += 6.0 * (P @ c)
    return -lap

  def grad_fun(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    grad = a[None, :] + 2.0 * P * b[None, :] + 3.0 * (P * P) * c[None, :]
    grad = grad.copy()
    for j in range(D):
      for i in range(D):
        if i != j:
          grad[:, j] += cross[i, j] * P[:, i]
    return grad

  return u_fun, f_fun, grad_fun


def affine_map_ref_to_phys(V_phys: np.ndarray, Xhat: np.ndarray) -> np.ndarray:
  V_phys = np.asarray(V_phys, dtype=np.float64)
  Xhat = np.asarray(Xhat, dtype=np.float64)
  B = V_phys[:, 1:] - V_phys[:, [0]]
  return np.ascontiguousarray(V_phys[:, 0][None, :] + Xhat @ B.T)


def canonical_face_bary(Y: np.ndarray) -> np.ndarray:
  Y = np.asarray(Y, dtype=np.float64)
  B = np.empty((Y.shape[0], Y.shape[1] + 1), dtype=np.float64)
  B[:, 0] = 1.0 - np.sum(Y, axis=1)
  B[:, 1:] = Y
  return B


def face_points_in_volume_ref(
  D: int,
  face_id: int,
  sigma_local_to_canonical: np.ndarray,
  Y: np.ndarray,
) -> np.ndarray:
  sigma = np.asarray(sigma_local_to_canonical, dtype=np.int64)
  if sigma.shape != (D,):
    raise ValueError(f"sigma must have shape ({D},)")

  B_can = canonical_face_bary(Y)
  lam_vol = np.zeros((B_can.shape[0], D + 1), dtype=np.float64)
  fv = face_vertices(D, face_id).astype(np.int64)
  for i_local in range(D):
    lam_vol[:, fv[i_local]] = B_can[:, int(sigma[i_local])]
  return np.ascontiguousarray(lam_vol[:, 1:])


def element_vertices(
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplex: np.ndarray,
) -> np.ndarray:
  row_of_id = {int(vid): i for i, vid in enumerate(vertex_ids)}
  rows = [row_of_id[int(vid)] for vid in simplex]
  return np.asfortranarray(coords[rows, :].T)


def physical_face_geometry(
  D: int,
  V_phys: np.ndarray,
  global_vids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Return physical face scales and outward unit normals in local face order."""
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  global_vids = np.asarray(global_vids, dtype=np.int32)
  nface = D + 1
  face_scale = np.empty(nface, dtype=np.float64)
  unit_normal = np.empty((nface, D), dtype=np.float64)

  if D == 1:
    x0 = float(V_phys[0, 0])
    x1 = float(V_phys[0, 1])
    sgn = 1.0 if x1 >= x0 else -1.0
    unit_normal[0, 0] = sgn
    unit_normal[1, 0] = -sgn
    face_scale[:] = 1.0
    return face_scale, unit_normal

  for face_id in range(nface):
    fv = face_vertices(D, face_id).astype(np.int64)
    sigma = face_sigma_array(global_vids, face_id).astype(np.int64)

    P_can = np.empty((D, D), dtype=np.float64)
    for i_local in range(D):
      P_can[:, int(sigma[i_local])] = V_phys[:, fv[i_local]]

    E = P_can[:, 1:] - P_can[:, [0]]
    face_scale[face_id] = math.sqrt(float(np.linalg.det(E.T @ E)))

    _, _, vh = np.linalg.svd(E.T, full_matrices=True)
    normal = vh[-1, :].copy()
    normal /= np.linalg.norm(normal)
    if float(np.dot(normal, V_phys[:, face_id] - P_can[:, 0])) > 0.0:
      normal *= -1.0
    unit_normal[face_id, :] = normal

  return face_scale, unit_normal


def find_local_face_id(simplex: np.ndarray, key: tuple[int, ...]) -> int:
  target = tuple(sorted(int(v) for v in key))
  for face_id in range(simplex.size):
    candidate = tuple(sorted(np.delete(simplex, face_id).astype(int).tolist()))
    if candidate == target:
      return face_id
  raise ValueError(f"face {key} is not a face of simplex {simplex.tolist()}")


def project_source_elementmajor(
  pc: RefSimplexPrecomp,
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplices: np.ndarray,
  f_fun: ScalarFun,
) -> np.ndarray:
  Xhat, What = pc.residual_quad()
  V_int = pc.residual_basis()[:, :pc.m_int]
  out = np.empty((simplices.shape[0], pc.m_int), dtype=np.float64)

  for e, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_ids, coords, simplex)
    B = V_phys[:, 1:] - V_phys[:, [0]]
    detBabs = abs(float(np.linalg.det(B)))
    P = affine_map_ref_to_phys(V_phys, Xhat)
    out[e, :] = detBabs * (V_int.T @ (What * f_fun(P)))

  return np.ascontiguousarray(out)


def project_robin_boundary_data(
  pc: RefSimplexPrecomp,
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplices: np.ndarray,
  face_to_elements: dict[tuple[int, ...], tuple[int, ...]],
  alpha: float,
  beta: float,
  u_fun: ScalarFun,
  grad_fun: GradFun,
) -> tuple[np.ndarray, np.ndarray]:
  """Project alpha*u + beta*du/dn in the canonical common-face basis."""
  Y, W_face, V_face = pc.face_quad_basis()
  boundary_items = sorted(
    (key, owners[0])
    for key, owners in face_to_elements.items()
    if len(owners) == 1
  )

  keys = np.empty((len(boundary_items), pc.D), dtype=np.int32)
  g = np.empty((len(boundary_items), pc.kf), dtype=np.float64)

  for j, (key, elem_id) in enumerate(boundary_items):
    simplex = simplices[int(elem_id)]
    face_id = find_local_face_id(simplex, key)
    V_phys = element_vertices(vertex_ids, coords, simplex)
    face_scale, unit_normal = physical_face_geometry(pc.D, V_phys, simplex)

    sigma = face_sigma_array(simplex, face_id).astype(np.int32)
    Xhat_face = face_points_in_volume_ref(pc.D, face_id, sigma, Y)
    P = affine_map_ref_to_phys(V_phys, Xhat_face)

    u_vals = u_fun(P)
    q_vals = grad_fun(P) @ unit_normal[face_id, :]
    robin_vals = float(alpha) * u_vals + float(beta) * q_vals

    keys[j, :] = np.asarray(key, dtype=np.int32)
    g[j, :] = V_face.T @ ((W_face * face_scale[face_id]) * robin_vals)

  return np.ascontiguousarray(keys), np.ascontiguousarray(g)


def project_exact_leaf_coeffs(
  pc_eval: RefSimplexPrecomp,
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplices: np.ndarray,
  u_fun: ScalarFun,
) -> np.ndarray:
  Xhat, What = pc_eval.volume_quad()
  V = pc_eval.volume_basis()
  out = np.empty((simplices.shape[0], pc_eval.M), dtype=np.float64)
  for e, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_ids, coords, simplex)
    P = affine_map_ref_to_phys(V_phys, Xhat)
    out[e, :] = V.T @ (What * u_fun(P))
  return out


def global_relative_L2_error(
  pc_eval: RefSimplexPrecomp,
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplices: np.ndarray,
  leaf_coeffs: np.ndarray,
  u_fun: ScalarFun,
) -> float:
  Xhat, What = pc_eval.volume_quad()
  V = pc_eval.volume_basis()
  err2 = 0.0
  ref2 = 0.0
  for e, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_ids, coords, simplex)
    B = V_phys[:, 1:] - V_phys[:, [0]]
    detBabs = abs(float(np.linalg.det(B)))
    P = affine_map_ref_to_phys(V_phys, Xhat)
    u_true = u_fun(P)
    u_num = V @ leaf_coeffs[e, :]
    err2 += detBabs * float(np.sum(What * (u_num - u_true) ** 2))
    ref2 += detBabs * float(np.sum(What * u_true ** 2))
  return math.sqrt(err2 / max(ref2, 1.0e-300))


def count_mesh_faces(mesh) -> tuple[int, int]:
  nboundary = sum(len(owners) == 1 for owners in mesh.face_to_elements.values())
  ninterior = sum(len(owners) == 2 for owners in mesh.face_to_elements.values())
  return int(nboundary), int(ninterior)


def run_dimension(
  D: int,
  args: argparse.Namespace,
  output_dir: Path,
) -> tuple[HpsPoissonResult, float, float]:
  mesh_rng = np.random.default_rng(args.mesh_seed + D)
  mesh = generate_mesh(
    D,
    DEFAULT_INTERIOR_POINT_COUNTS[D],
    mesh_rng,
    DEFAULT_QHULL_OPTIONS,
    args.determinant_tol,
  )
  tree = build_merge_tree(
    mesh.adjacency,
    partitioner="pymetis",
    seed=args.tree_seed + D,
  )
  validate_merge_tree(mesh.adjacency, tree.merge_pairs, tree.root_id)

  vertex_ids = np.arange(mesh.X.shape[0], dtype=np.int32)
  kappa = np.array([0.73 + 0.19 * i for i in range(D + 1)], dtype=np.float64)
  q_vol = args.n + args.q_pad
  q_face = 1 if D == 1 else args.n + args.q_pad
  pc = RefSimplexPrecomp(
    D, args.n, kappa,
    q_pad=args.q_pad,
    q_vol=q_vol,
    q_face=q_face,
  )

  u_fun, f_fun, grad_fun = manufactured_polynomial(D)
  f_int = project_source_elementmajor(
    pc, vertex_ids, mesh.X, mesh.simplices, f_fun
  )
  boundary_keys, boundary_g = project_robin_boundary_data(
    pc,
    vertex_ids,
    mesh.X,
    mesh.simplices,
    mesh.face_to_elements,
    args.alpha,
    args.beta,
    u_fun,
    grad_fun,
  )

  result = run_poisson_mesh_tree_solve(
    pc,
    vertex_ids,
    mesh.X,
    mesh.simplices,
    tree.merge_pairs,
    f_int,
    boundary_keys,
    boundary_g,
    tau_C=args.tau_C,
    alpha=args.alpha,
    beta=args.beta,
    verbose=args.verbose,
  )

  q_eval = args.n + args.q_eval_pad
  pc_eval = RefSimplexPrecomp(
    D, args.n, kappa,
    q_pad=args.q_eval_pad,
    q_vol=q_eval,
    q_face=1 if D == 1 else q_eval,
  )
  exact_coeffs = project_exact_leaf_coeffs(
    pc_eval, vertex_ids, mesh.X, mesh.simplices, u_fun
  )
  rel_coef = float(
    np.linalg.norm(result.leaf_coeffs - exact_coeffs)
    / max(np.linalg.norm(exact_coeffs), 1.0e-300)
  )
  rel_L2 = global_relative_L2_error(
    pc_eval, vertex_ids, mesh.X, mesh.simplices, result.leaf_coeffs, u_fun
  )

  nboundary, ninterior = count_mesh_faces(mesh)
  residuals = np.array([
    result.root_robin_residual_inf,
    result.interface_flux_residual_inf,
    result.parent_consistency_residual_inf,
  ])
  assert np.all(np.isfinite(residuals)), result
  assert np.max(residuals) < args.residual_tol, result
  assert rel_coef < args.solution_tol, (D, rel_coef, result)
  assert rel_L2 < args.solution_tol, (D, rel_L2, result)
  assert result.root_nb == nboundary * pc.kf, result
  assert result.interface_nb == ninterior * pc.kf, result

  mesh_file = export_case(output_dir, mesh, tree)
  figure_file = visualize_case(output_dir, mesh, tree, show=False)

  print(
    f"D={D}: nverts={mesh.X.shape[0]} nelem={mesh.simplices.shape[0]} "
    f"M={pc.M} m_int={pc.m_int} kf={pc.kf} "
    f"boundary_faces={nboundary} interior_faces={ninterior} "
    f"merges={tree.merge_pairs.shape[0]} depth={tree.max_depth} "
    f"metis={tree.metis_splits} fallback={tree.fallback_splits}"
  )
  print(
    f"  alpha={args.alpha:g} beta={args.beta:g} "
    f"root_res={result.root_robin_residual_inf:.3e} "
    f"iface_res={result.interface_flux_residual_inf:.3e} "
    f"parent_res={result.parent_consistency_residual_inf:.3e}"
  )
  print(f"  rel_coef={rel_coef:.3e} rel_L2={rel_L2:.3e}")
  print(f"  mesh/tree data: {mesh_file}")
  if figure_file is not None:
    print(f"  visualization: {figure_file}")

  return result, rel_coef, rel_L2


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Polynomial manufactured Poisson/Robin HPS test on external SciPy/PyMetis meshes."
  )
  parser.add_argument("--D", default="1,2,3")
  parser.add_argument("--n", type=int, default=3)
  parser.add_argument("--q-pad", type=int, default=1)
  parser.add_argument("--q-eval-pad", type=int, default=1)
  parser.add_argument("--alpha", type=float, default=1.0)
  parser.add_argument("--beta", type=float, default=0.2)
  parser.add_argument("--tau-C", type=float, default=10.0)
  parser.add_argument("--residual-tol", type=float, default=5.0e-8)
  parser.add_argument("--solution-tol", type=float, default=2.0e-7)
  parser.add_argument("--determinant-tol", type=float, default=1.0e-13)
  parser.add_argument("--mesh-seed", type=int, default=41000)
  parser.add_argument("--tree-seed", type=int, default=42000)
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  if args.n < 3:
    raise ValueError("this manufactured solution is cubic, so use n >= 3")
  if args.alpha == 0.0:
    raise ValueError("pure Neumann is not yet supported; use alpha != 0")

  try:
    import pymetis
  except ImportError as exc:
    raise RuntimeError("this test requires PyMetis") from exc
  try:
    version = importlib.metadata.version("pymetis")
  except importlib.metadata.PackageNotFoundError:
    version = "unknown"
  print(
    "mesh-tree partitioner: pymetis "
    f"version={version} module={getattr(pymetis, '__file__', '<unknown>')}; "
    "validated fallback enabled"
  )

  args.output_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for D in parse_D_list(args.D):
    rows.append(run_dimension(D, args, args.output_dir))

  print("\nall requested external-mesh Poisson HPS manufactured tests passed")
  for result, rel_coef, rel_L2 in rows:
    print(
      f"D={result.D} nelem={result.nelem} M={result.M} "
      f"rel_coef={rel_coef:.3e} rel_L2={rel_L2:.3e}"
    )


if __name__ == "__main__":
  main()
