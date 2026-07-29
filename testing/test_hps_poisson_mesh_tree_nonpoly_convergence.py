from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

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
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "build" / "hps_poisson_nonpoly_convergence"

# Keep one fixed mesh/tree for every polynomial degree in each dimension.
# D=4,5 are intentionally kept at the cube-corner triangulation because the
# cost of materializing all leaf trace columns grows rapidly with n and D.
DEFAULT_INTERIOR_POINT_COUNTS = {
  1: 4,
  2: 1,
  3: 0,
  4: 0,
  5: 0,
}

ScalarPointFun = Callable[..., np.ndarray]
GradPointFun = Callable[..., tuple[np.ndarray, ...]]


def dimPi(D: int, n: int) -> int:
  return math.comb(n + D, D) if n >= 0 else 0


def parse_D_list(text: str) -> list[int]:
  if text.strip().lower() == "all":
    return [1, 2, 3, 4, 5]
  out = [int(part.strip()) for part in text.split(",") if part.strip()]
  if not out or any(D < 1 or D > 5 for D in out):
    raise ValueError("--D must list dimensions in 1..5 or be 'all'")
  return out


def make_nonpoly_u_f_grad_D(
  D: int,
  simplify: bool = False,
) -> tuple[ScalarPointFun, ScalarPointFun, GradPointFun]:
  """Build numpy-callable u, f=-Delta u, and grad u in D variables."""
  xs = sp.symbols(f"x0:{D}", real=True)

  lin = sum(sp.Rational(i + 2, 7) * xs[i] for i in range(D))
  quad = sum(sp.Rational(i + 1, 11) * xs[i] ** 2 for i in range(D))
  mix = 0
  for i in range(D):
    for j in range(i + 1, D):
      mix += sp.Rational(1, 13 + i + 2 * j) * xs[i] * xs[j]

  u_expr = (
    sp.exp(sp.Rational(1, 5) * sp.cos(lin + sp.Rational(1, 3) * quad))
    + sp.sin(lin + mix)
    + sp.Rational(1, 10) * sp.exp(-quad)
  )

  grad_expr = [sp.diff(u_expr, x) for x in xs]
  lap_expr = sum(sp.diff(u_expr, x, 2) for x in xs)
  f_expr = -lap_expr

  if simplify:
    grad_expr = [sp.simplify(g) for g in grad_expr]
    f_expr = sp.simplify(f_expr)

  u_fun = sp.lambdify(xs, u_expr, "numpy")
  f_fun = sp.lambdify(xs, f_expr, "numpy")
  grad_funs = [sp.lambdify(xs, g, "numpy") for g in grad_expr]

  def grad_fun(*args):
    return tuple(g(*args) for g in grad_funs)

  return u_fun, f_fun, grad_fun


def eval_scalar_D(fun: ScalarPointFun, P: np.ndarray) -> np.ndarray:
  P = np.asarray(P, dtype=np.float64)
  D = P.shape[1]
  vals = fun(*[P[:, i] for i in range(D)])
  vals = np.asarray(vals, dtype=np.float64)
  if vals.shape == ():
    vals = np.full(P.shape[0], float(vals), dtype=np.float64)
  return np.ravel(vals).astype(np.float64, copy=False)


def eval_grad_D(grad_fun: GradPointFun, P: np.ndarray) -> np.ndarray:
  P = np.asarray(P, dtype=np.float64)
  D = P.shape[1]
  parts = grad_fun(*[P[:, i] for i in range(D)])
  if len(parts) != D:
    raise ValueError(f"grad_fun returned {len(parts)} components, expected {D}")

  out = np.empty((P.shape[0], D), dtype=np.float64)
  for j, values in enumerate(parts):
    values = np.asarray(values, dtype=np.float64)
    if values.shape == ():
      out[:, j] = float(values)
    else:
      out[:, j] = np.ravel(values)
  return out


def affine_map_ref_to_phys(V_phys: np.ndarray, Xhat: np.ndarray) -> np.ndarray:
  V_phys = np.asarray(V_phys, dtype=np.float64)
  Xhat = np.asarray(Xhat, dtype=np.float64)
  B = V_phys[:, 1:] - V_phys[:, [0]]
  return np.ascontiguousarray(V_phys[:, 0][None, :] + Xhat @ B.T)


def canonical_face_bary(Y: np.ndarray) -> np.ndarray:
  Y = np.asarray(Y, dtype=np.float64)
  bary = np.empty((Y.shape[0], Y.shape[1] + 1), dtype=np.float64)
  bary[:, 0] = 1.0 - np.sum(Y, axis=1)
  bary[:, 1:] = Y
  return bary


def face_points_in_volume_ref(
  D: int,
  face_id: int,
  sigma_local_to_canonical: np.ndarray,
  Y: np.ndarray,
) -> np.ndarray:
  sigma = np.asarray(sigma_local_to_canonical, dtype=np.int64)
  if sigma.shape != (D,):
    raise ValueError(f"sigma must have shape ({D},)")

  bary_can = canonical_face_bary(Y)
  bary_vol = np.zeros((bary_can.shape[0], D + 1), dtype=np.float64)
  local_face_vertices = face_vertices(D, face_id).astype(np.int64)
  for i_local in range(D):
    bary_vol[:, local_face_vertices[i_local]] = bary_can[:, int(sigma[i_local])]
  return np.ascontiguousarray(bary_vol[:, 1:])


def make_vertex_row_map(vertex_ids: np.ndarray) -> dict[int, int]:
  return {int(vertex_id): row for row, vertex_id in enumerate(vertex_ids)}


def element_vertices(
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplex: np.ndarray,
) -> np.ndarray:
  rows = [vertex_row[int(vertex_id)] for vertex_id in simplex]
  return np.asfortranarray(coords[rows, :].T)


def physical_face_geometry(
  D: int,
  V_phys: np.ndarray,
  global_vids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Return physical face scales and outward unit normals in local order."""
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  global_vids = np.asarray(global_vids, dtype=np.int32)
  face_scale = np.empty(D + 1, dtype=np.float64)
  unit_normal = np.empty((D + 1, D), dtype=np.float64)

  if D == 1:
    x0 = float(V_phys[0, 0])
    x1 = float(V_phys[0, 1])
    sign = 1.0 if x1 >= x0 else -1.0
    unit_normal[0, 0] = sign
    unit_normal[1, 0] = -sign
    face_scale[:] = 1.0
    return face_scale, unit_normal

  for face_id in range(D + 1):
    local_face_vertices = face_vertices(D, face_id).astype(np.int64)
    sigma = face_sigma_array(global_vids, face_id).astype(np.int64)

    P_can = np.empty((D, D), dtype=np.float64)
    for i_local in range(D):
      P_can[:, int(sigma[i_local])] = V_phys[:, local_face_vertices[i_local]]

    edge_matrix = P_can[:, 1:] - P_can[:, [0]]
    face_scale[face_id] = math.sqrt(
      max(0.0, float(np.linalg.det(edge_matrix.T @ edge_matrix)))
    )

    _, _, vh = np.linalg.svd(edge_matrix.T, full_matrices=True)
    normal = vh[-1, :].copy()
    normal /= np.linalg.norm(normal)
    if float(np.dot(normal, V_phys[:, face_id] - P_can[:, 0])) > 0.0:
      normal *= -1.0
    unit_normal[face_id, :] = normal

  return face_scale, unit_normal


def find_local_face_id(simplex: np.ndarray, key: tuple[int, ...]) -> int:
  target = tuple(sorted(int(value) for value in key))
  for face_id in range(simplex.size):
    candidate = tuple(sorted(np.delete(simplex, face_id).astype(int).tolist()))
    if candidate == target:
      return face_id
  raise ValueError(f"face {key} is not a face of simplex {simplex.tolist()}")


def project_source_elementmajor(
  pc: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  f_fun: ScalarPointFun,
) -> np.ndarray:
  Xhat, weights = pc.residual_quad()
  V_int = pc.residual_basis()[:, :pc.m_int]
  out = np.empty((simplices.shape[0], pc.m_int), dtype=np.float64)

  for element_id, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_row, coords, simplex)
    B = V_phys[:, 1:] - V_phys[:, [0]]
    detBabs = abs(float(np.linalg.det(B)))
    points = affine_map_ref_to_phys(V_phys, Xhat)
    values = eval_scalar_D(f_fun, points)
    out[element_id, :] = detBabs * (V_int.T @ (weights * values))

  return np.ascontiguousarray(out)


def project_robin_boundary_data(
  pc: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  face_to_elements: dict[tuple[int, ...], tuple[int, ...]],
  alpha: float,
  beta: float,
  u_fun: ScalarPointFun,
  grad_fun: GradPointFun,
) -> tuple[np.ndarray, np.ndarray]:
  """Project alpha*u + beta*du/dn in the canonical common-face basis."""
  Y, weights, V_face = pc.face_quad_basis()
  boundary_items = sorted(
    (key, owners[0])
    for key, owners in face_to_elements.items()
    if len(owners) == 1
  )

  keys = np.empty((len(boundary_items), pc.D), dtype=np.int32)
  projected_g = np.empty((len(boundary_items), pc.kf), dtype=np.float64)

  for row, (key, element_id) in enumerate(boundary_items):
    simplex = simplices[int(element_id)]
    face_id = find_local_face_id(simplex, key)
    V_phys = element_vertices(vertex_row, coords, simplex)
    face_scale, unit_normal = physical_face_geometry(pc.D, V_phys, simplex)

    sigma = face_sigma_array(simplex, face_id).astype(np.int32)
    Xhat_face = face_points_in_volume_ref(pc.D, face_id, sigma, Y)
    points = affine_map_ref_to_phys(V_phys, Xhat_face)

    u_values = eval_scalar_D(u_fun, points)
    grad_values = eval_grad_D(grad_fun, points)
    normal_derivative = grad_values @ unit_normal[face_id, :]
    robin_values = float(alpha) * u_values + float(beta) * normal_derivative

    keys[row, :] = np.asarray(key, dtype=np.int32)
    projected_g[row, :] = V_face.T @ (
      (weights * face_scale[face_id]) * robin_values
    )

  return np.ascontiguousarray(keys), np.ascontiguousarray(projected_g)


def global_relative_L2_error(
  pc_eval: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  leaf_coeffs: np.ndarray,
  u_fun: ScalarPointFun,
) -> float:
  Xhat, weights = pc_eval.volume_quad()
  V_eval = pc_eval.volume_basis()
  error_squared = 0.0
  reference_squared = 0.0

  for element_id, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_row, coords, simplex)
    B = V_phys[:, 1:] - V_phys[:, [0]]
    detBabs = abs(float(np.linalg.det(B)))
    points = affine_map_ref_to_phys(V_phys, Xhat)
    u_true = eval_scalar_D(u_fun, points)
    u_numerical = V_eval @ leaf_coeffs[element_id, :]
    error_squared += detBabs * float(
      np.sum(weights * (u_numerical - u_true) ** 2)
    )
    reference_squared += detBabs * float(np.sum(weights * u_true ** 2))

  return math.sqrt(error_squared / max(reference_squared, 1.0e-300))


def count_mesh_faces(mesh) -> tuple[int, int]:
  boundary = sum(len(owners) == 1 for owners in mesh.face_to_elements.values())
  interior = sum(len(owners) == 2 for owners in mesh.face_to_elements.values())
  return int(boundary), int(interior)


def assert_hps_result(
  result: HpsPoissonResult,
  *,
  boundary_faces: int,
  interior_faces: int,
  residual_tol: float,
) -> None:
  residuals = np.asarray([
    result.root_robin_residual_inf,
    result.interface_flux_residual_inf,
    result.parent_consistency_residual_inf,
  ])
  assert np.all(np.isfinite(residuals)), result
  assert np.max(residuals) < residual_tol, result
  assert np.all(np.isfinite(result.leaf_coeffs)), result
  assert result.root_nb == boundary_faces * result.kf, result
  assert result.interface_nb == interior_faces * result.kf, result


def save_results_csv(rows: list[dict[str, float | int]], path: Path) -> None:
  fieldnames = [
    "D", "n", "dimPi", "nverts", "nelem", "M", "m_int", "kf",
    "boundary_faces", "interior_faces", "tree_depth", "metis_splits",
    "fallback_splits", "alpha", "beta", "tau_C", "q_vol", "q_face",
    "q_eval", "root_residual", "interface_residual", "parent_residual",
    "relative_L2_error",
  ]
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def plot_l2_vs_n(rows: list[dict[str, float | int]], path: Path) -> None:
  plt.figure(figsize=(7.5, 5.2))
  for D in sorted({int(row["D"]) for row in rows}):
    subset = sorted(
      (row for row in rows if int(row["D"]) == D),
      key=lambda row: int(row["n"]),
    )
    x = np.asarray([int(row["n"]) for row in subset], dtype=np.int64)
    y = np.asarray([float(row["relative_L2_error"]) for row in subset])
    plt.semilogy(x, y, marker="o", label=f"D={D}")

  plt.xlabel(r"polynomial degree $n$")
  plt.ylabel(r"relative $L^2$ error")
  plt.title("HPS Poisson nonpolynomial convergence")
  plt.xticks(sorted({int(row["n"]) for row in rows}))
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def plot_l2_vs_dimPi(rows: list[dict[str, float | int]], path: Path) -> None:
  plt.figure(figsize=(7.5, 5.2))
  for D in sorted({int(row["D"]) for row in rows}):
    subset = sorted(
      (row for row in rows if int(row["D"]) == D),
      key=lambda row: int(row["dimPi"]),
    )
    x = np.asarray([int(row["dimPi"]) for row in subset], dtype=np.int64)
    y = np.asarray([float(row["relative_L2_error"]) for row in subset])
    plt.semilogy(x, y, marker="o", label=f"D={D}")

  plt.xlabel(r"$\dim(\Pi_n^D)$")
  plt.ylabel(r"relative $L^2$ error")
  plt.title("HPS Poisson nonpolynomial convergence by approximation dimension")
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Nonpolynomial manufactured Poisson/Robin convergence study on fixed "
      "external SciPy/PyMetis meshes for D=1..5."
    )
  )
  parser.add_argument("--D", default="all")
  parser.add_argument("--min-n", type=int, default=2)
  parser.add_argument("--max-n", type=int, default=6)
  parser.add_argument("--q-pad", type=int, default=1)
  parser.add_argument("--q-eval-pad", type=int, default=1)
  parser.add_argument("--alpha", type=float, default=1.0)
  parser.add_argument("--beta", type=float, default=2.3)
  parser.add_argument("--tau-C", type=float, default=10.0)
  parser.add_argument("--residual-tol", type=float, default=5.0e-8)
  parser.add_argument("--determinant-tol", type=float, default=1.0e-13)
  parser.add_argument("--mesh-seed", type=int, default=51000)
  parser.add_argument("--tree-seed", type=int, default=52000)
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument("--simplify-symbolics", action="store_true")
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  if args.min_n < 2:
    raise ValueError(
      "the current C++ Poisson HPS backend requires n >= 2; use --min-n 2"
    )
  if args.max_n < args.min_n:
    raise ValueError("--max-n must be at least --min-n")
  if args.alpha == 0.0:
    raise ValueError("pure Neumann is not yet supported; use alpha != 0")
  if args.q_pad < 0 or args.q_eval_pad < 0:
    raise ValueError("quadrature padding values must be nonnegative")

  try:
    import pymetis
  except ImportError as exc:
    raise RuntimeError("this test requires PyMetis") from exc
  try:
    pymetis_version = importlib.metadata.version("pymetis")
  except importlib.metadata.PackageNotFoundError:
    pymetis_version = "unknown"
  print(
    "mesh-tree partitioner: pymetis "
    f"version={pymetis_version} "
    f"module={getattr(pymetis, '__file__', '<unknown>')}; "
    "validated fallback enabled"
  )

  args.output_dir.mkdir(parents=True, exist_ok=True)
  dimensions = parse_D_list(args.D)
  rows: list[dict[str, float | int]] = []

  for D in dimensions:
    print(f"\nConstructing fixed mesh/tree and symbolic manufactured solution for D={D}")
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

    mesh_file = export_case(args.output_dir, mesh, tree)
    figure_file = visualize_case(args.output_dir, mesh, tree, show=False)
    vertex_ids = np.arange(mesh.X.shape[0], dtype=np.int32)
    vertex_row = make_vertex_row_map(vertex_ids)
    boundary_faces, interior_faces = count_mesh_faces(mesh)
    u_fun, f_fun, grad_fun = make_nonpoly_u_f_grad_D(
      D,
      simplify=args.simplify_symbolics,
    )
    kappa = np.asarray(
      [0.73 + 0.19 * i for i in range(D + 1)],
      dtype=np.float64,
    )

    print(
      f"D={D}: nverts={mesh.X.shape[0]} nelem={mesh.simplices.shape[0]} "
      f"boundary_faces={boundary_faces} interior_faces={interior_faces} "
      f"merges={tree.merge_pairs.shape[0]} depth={tree.max_depth} "
      f"metis={tree.metis_splits} fallback={tree.fallback_splits}"
    )
    print(f"  mesh/tree data: {mesh_file}")
    if figure_file is not None:
      print(f"  visualization: {figure_file}")

    for n in range(args.min_n, args.max_n + 1):
      q_vol = n + args.q_pad
      q_face = 1 if D == 1 else n + args.q_pad
      pc = RefSimplexPrecomp(
        D,
        n,
        kappa,
        q_pad=args.q_pad,
        q_vol=q_vol,
        q_face=q_face,
      )

      f_int = project_source_elementmajor(
        pc,
        vertex_row,
        mesh.X,
        mesh.simplices,
        f_fun,
      )
      boundary_keys, boundary_g = project_robin_boundary_data(
        pc,
        vertex_row,
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
      assert_hps_result(
        result,
        boundary_faces=boundary_faces,
        interior_faces=interior_faces,
        residual_tol=args.residual_tol,
      )

      q_eval = n + args.q_eval_pad
      pc_eval = RefSimplexPrecomp(
        D,
        n,
        kappa,
        q_pad=args.q_eval_pad,
        q_vol=q_eval,
        q_face=1 if D == 1 else q_eval,
      )
      relative_l2 = global_relative_L2_error(
        pc_eval,
        vertex_row,
        mesh.X,
        mesh.simplices,
        result.leaf_coeffs,
        u_fun,
      )
      if not np.isfinite(relative_l2):
        raise AssertionError(f"non-finite relative L2 error for D={D}, n={n}")

      row: dict[str, float | int] = {
        "D": D,
        "n": n,
        "dimPi": dimPi(D, n),
        "nverts": int(mesh.X.shape[0]),
        "nelem": int(mesh.simplices.shape[0]),
        "M": result.M,
        "m_int": result.m_int,
        "kf": result.kf,
        "boundary_faces": boundary_faces,
        "interior_faces": interior_faces,
        "tree_depth": tree.max_depth,
        "metis_splits": tree.metis_splits,
        "fallback_splits": tree.fallback_splits,
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "tau_C": float(args.tau_C),
        "q_vol": q_vol,
        "q_face": q_face,
        "q_eval": q_eval,
        "root_residual": result.root_robin_residual_inf,
        "interface_residual": result.interface_flux_residual_inf,
        "parent_residual": result.parent_consistency_residual_inf,
        "relative_L2_error": relative_l2,
      }
      rows.append(row)

      print(
        f"  n={n:2d} dimPi={row['dimPi']:5d} M={result.M:5d} "
        f"m_int={result.m_int:5d} kf={result.kf:5d} "
        f"rel_L2={relative_l2:.3e} "
        f"root={result.root_robin_residual_inf:.3e} "
        f"iface={result.interface_flux_residual_inf:.3e} "
        f"parent={result.parent_consistency_residual_inf:.3e}"
      )

  csv_path = args.output_dir / "hps_poisson_nonpoly_convergence.csv"
  plot_n_path = args.output_dir / "hps_poisson_nonpoly_l2_vs_n.png"
  plot_dim_path = args.output_dir / "hps_poisson_nonpoly_l2_vs_dimPi.png"
  save_results_csv(rows, csv_path)
  plot_l2_vs_n(rows, plot_n_path)
  plot_l2_vs_dimPi(rows, plot_dim_path)

  print("\nall requested external-mesh nonpolynomial Poisson HPS solves completed")
  print(f"results CSV: {csv_path}")
  print(f"L2 versus n plot: {plot_n_path}")
  print(f"L2 versus dimPi plot: {plot_dim_path}")


if __name__ == "__main__":
  main()
