from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from hps_mesh_tree_driver import (
  DEFAULT_QHULL_OPTIONS,
  build_merge_tree,
  export_case,
  generate_mesh,
  validate_merge_tree,
  visualize_case,
)
from jhps import (
  HpsEllipticResult,
  HpsLeafOperatorMode,
  run_elliptic_mesh_tree_solve,
)
from jperms import face_sigma_array, face_vertices
from jprecomp import RefSimplexPrecomp, dimPi
from thread_control import set_omp_threads, set_openblas_threads


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "python" else _THIS.parent
_DEFAULT_OUTPUT_DIR = (
  _PROJECT_ROOT
  / "build"
  / "hps_elliptic_nonpoly_dense_sparse_convergence"
)

# Keep one fixed mesh and merge tree for every polynomial degree in each
# dimension. Higher-dimensional cases intentionally use only cube corners.
#DEFAULT_INTERIOR_POINT_COUNTS = {
#  1: 4,
#  2: 1,
#  3: 0,
#  4: 0,
#  5: 0,
#}
#DEFAULT_INTERIOR_POINT_COUNTS = {
#  1: 2,
#  2: 12,
#  3: 6,
#  4: 2,
#  5: 1,
#}
DEFAULT_INTERIOR_POINT_COUNTS = {
  1: 0,
  2: 0,
  3: 1,
  4: 1,
  5: 1,
}

ScalarPointFun = Callable[..., np.ndarray]
GradPointFun = Callable[..., tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class SymbolicProblem:
  symbols: tuple[sp.Symbol, ...]
  u_expr: sp.Expr
  A_expr: tuple[tuple[sp.Expr, ...], ...]
  b_expr: tuple[sp.Expr, ...]
  c_expr: sp.Expr
  Lu_expr: sp.Expr
  f_expr: sp.Expr
  u_fun: ScalarPointFun
  grad_fun: GradPointFun
  A_fun: tuple[tuple[ScalarPointFun, ...], ...]
  b_fun: tuple[ScalarPointFun, ...]
  c_fun: ScalarPointFun
  f_fun: ScalarPointFun


def parse_D_list(text: str) -> list[int]:
  if text.strip().lower() == "all":
    return [1, 2, 3, 4, 5]
  out = [int(part.strip()) for part in text.split(",") if part.strip()]
  if not out or any(D < 1 or D > 5 for D in out):
    raise ValueError("--D must list dimensions in 1..5 or be 'all'")
  return out



def make_scalar_fun(
  expr: sp.Expr,
  symbols: tuple[sp.Symbol, ...],
) -> ScalarPointFun:
  raw = sp.lambdify(symbols, expr, modules="numpy")

  def wrapped(*args):
    values = np.asarray(raw(*args), dtype=np.float64)
    if values.ndim == 0:
      first = np.asarray(args[0])
      values = np.full(first.shape, float(values), dtype=np.float64)
    return values

  return wrapped


def make_grad_fun(
  expressions: tuple[sp.Expr, ...],
  symbols: tuple[sp.Symbol, ...],
) -> GradPointFun:
  raw = tuple(sp.lambdify(symbols, expr, modules="numpy") for expr in expressions)

  def wrapped(*args):
    out = []
    first = np.asarray(args[0])
    for fun in raw:
      values = np.asarray(fun(*args), dtype=np.float64)
      if values.ndim == 0:
        values = np.full(first.shape, float(values), dtype=np.float64)
      out.append(values)
    return tuple(out)

  return wrapped


def build_symbolic_problem(D: int, simplify: bool = False) -> SymbolicProblem:
  """Build a globally smooth nondivergence-form manufactured problem.

  The principal tensor is symmetric and uniformly positive definite. The
  zero-order coefficient is negative, which keeps the Dirichlet problem away
  from a Helmholtz-type resonance for the sign convention used by the leaf.
  """
  symbols = tuple(sp.symbols(f"x0:{D}", real=True))
  s = sum(symbols)
  weighted_s = sum((r + 1) * symbols[r] for r in range(D))

  lin = sum(sp.Rational(r + 2, 7) * symbols[r] for r in range(D))
  quad = sum(sp.Rational(r + 1, 11) * symbols[r] ** 2 for r in range(D))
  mix = sum(
    sp.Rational(1, 13 + r + 2 * t) * symbols[r] * symbols[t]
    for r in range(D)
    for t in range(r + 1, D)
  )
  u_expr = (
    sp.exp(sp.Rational(1, 5) * sp.cos(lin + sp.Rational(1, 3) * quad))
    + sp.sin(lin + mix)
    + sp.Rational(1, 10) * sp.exp(-quad)
  )

  A = [[sp.Integer(0) for _ in range(D)] for _ in range(D)]
  for r in range(D):
    A[r][r] = (
      sp.Rational(3, 2)
      + sp.Rational(1, 10) * sp.sin(s + symbols[r])
    )
  for r in range(D):
    for t in range(r + 1, D):
      value = sp.Rational(1, 50) * sp.cos(symbols[r] - symbols[t])
      A[r][t] = value
      A[t][r] = value

  b = tuple(
    sp.Rational(1, 10) * sp.sin(s + (r + 1) * symbols[r])
    for r in range(D)
  )
  c_expr = -sp.Rational(2, 5) - sp.Rational(1, 10) * sp.cos(s)

  principal_expr = sum(
    A[r][t] * sp.diff(u_expr, symbols[r], symbols[t])
    for r in range(D)
    for t in range(D)
  )
  first_expr = sum(
    b[r] * sp.diff(u_expr, symbols[r])
    for r in range(D)
  )
  Lu_expr = principal_expr + first_expr + c_expr * u_expr
  f_expr = -Lu_expr
  grad_expr = tuple(sp.diff(u_expr, symbol) for symbol in symbols)

  if simplify:
    Lu_expr = sp.simplify(Lu_expr)
    f_expr = sp.simplify(f_expr)
    grad_expr = tuple(sp.simplify(expr) for expr in grad_expr)

  A_expr = tuple(tuple(A[r][t] for t in range(D)) for r in range(D))
  A_fun = tuple(
    tuple(make_scalar_fun(A_expr[r][t], symbols) for t in range(D))
    for r in range(D)
  )
  b_fun = tuple(make_scalar_fun(expr, symbols) for expr in b)

  return SymbolicProblem(
    symbols=symbols,
    u_expr=u_expr,
    A_expr=A_expr,
    b_expr=b,
    c_expr=c_expr,
    Lu_expr=Lu_expr,
    f_expr=f_expr,
    u_fun=make_scalar_fun(u_expr, symbols),
    grad_fun=make_grad_fun(grad_expr, symbols),
    A_fun=A_fun,
    b_fun=b_fun,
    c_fun=make_scalar_fun(c_expr, symbols),
    f_fun=make_scalar_fun(f_expr, symbols),
  )


def eval_scalar_D(fun: ScalarPointFun, P: np.ndarray) -> np.ndarray:
  P = np.asarray(P, dtype=np.float64)
  values = fun(*[P[:, i] for i in range(P.shape[1])])
  values = np.asarray(values, dtype=np.float64)
  if values.shape == ():
    values = np.full(P.shape[0], float(values), dtype=np.float64)
  return np.ravel(values).astype(np.float64, copy=False)


def eval_grad_D(grad_fun: GradPointFun, P: np.ndarray) -> np.ndarray:
  P = np.asarray(P, dtype=np.float64)
  D = P.shape[1]
  parts = grad_fun(*[P[:, i] for i in range(D)])
  if len(parts) != D:
    raise ValueError(f"grad_fun returned {len(parts)} components, expected {D}")

  out = np.empty((P.shape[0], D), dtype=np.float64)
  for r, values in enumerate(parts):
    values = np.asarray(values, dtype=np.float64)
    if values.shape == ():
      out[:, r] = float(values)
    else:
      out[:, r] = np.ravel(values)
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


def project_coefficient_fields_elementmajor(
  pc_data: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  problem: SymbolicProblem,
  p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Project A, b, and c in the common residual Jacobi family."""
  Xhat, weights, V_res = pc_data.residual_quad_basis()
  Mp = dimPi(pc_data.D, p)
  Vp = V_res[:, :Mp]
  nelem = simplices.shape[0]

  A = np.empty((nelem, pc_data.D, pc_data.D, Mp), dtype=np.float64)
  b = np.empty((nelem, pc_data.D, Mp), dtype=np.float64)
  c = np.empty((nelem, Mp), dtype=np.float64)

  for element_id, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_row, coords, simplex)
    points = affine_map_ref_to_phys(V_phys, Xhat)

    for r in range(pc_data.D):
      for t in range(pc_data.D):
        values = eval_scalar_D(problem.A_fun[r][t], points)
        A[element_id, r, t, :] = Vp.T @ (weights * values)

    for r in range(pc_data.D):
      values = eval_scalar_D(problem.b_fun[r], points)
      b[element_id, r, :] = Vp.T @ (weights * values)

    c_values = eval_scalar_D(problem.c_fun, points)
    c[element_id, :] = Vp.T @ (weights * c_values)

  return (
    np.ascontiguousarray(A),
    np.ascontiguousarray(b),
    np.ascontiguousarray(c),
  )


def project_source_elementmajor(
  pc_data: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  f_fun: ScalarPointFun,
) -> np.ndarray:
  """Project f for the leaf convention L u + f = 0."""
  Xhat, weights, V_res = pc_data.residual_quad_basis()
  V_int = V_res[:, :pc_data.M]
  out = np.empty((simplices.shape[0], pc_data.M), dtype=np.float64)

  for element_id, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_row, coords, simplex)
    B = V_phys[:, 1:] - V_phys[:, [0]]
    detBabs = abs(float(np.linalg.det(B)))
    points = affine_map_ref_to_phys(V_phys, Xhat)
    values = eval_scalar_D(f_fun, points)
    out[element_id, :] = detBabs * (V_int.T @ (weights * values))

  return np.ascontiguousarray(out)


def project_robin_boundary_data(
  pc_data: RefSimplexPrecomp,
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
  Y, weights, V_face = pc_data.face_quad_basis()
  boundary_items = sorted(
    (key, owners[0])
    for key, owners in face_to_elements.items()
    if len(owners) == 1
  )

  keys = np.empty((len(boundary_items), pc_data.D), dtype=np.int32)
  projected_g = np.empty((len(boundary_items), pc_data.kf), dtype=np.float64)

  for row, (key, element_id) in enumerate(boundary_items):
    simplex = simplices[int(element_id)]
    face_id = find_local_face_id(simplex, key)
    V_phys = element_vertices(vertex_row, coords, simplex)
    face_scale, unit_normal = physical_face_geometry(
      pc_data.D,
      V_phys,
      simplex,
    )

    sigma = face_sigma_array(simplex, face_id).astype(np.int32)
    Xhat_face = face_points_in_volume_ref(pc_data.D, face_id, sigma, Y)
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


def global_relative_L2_errors(
  pc_eval: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  leaf_coeffs: np.ndarray,
  u_fun: ScalarPointFun,
) -> tuple[float, float]:
  """Return HPS and best discontinuous elementwise projection errors."""
  Xhat, weights = pc_eval.volume_quad()
  V_eval = pc_eval.volume_basis()
  hps_error_squared = 0.0
  best_error_squared = 0.0
  reference_squared = 0.0

  for element_id, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_row, coords, simplex)
    B = V_phys[:, 1:] - V_phys[:, [0]]
    detBabs = abs(float(np.linalg.det(B)))
    points = affine_map_ref_to_phys(V_phys, Xhat)
    u_true = eval_scalar_D(u_fun, points)

    u_hps = V_eval @ leaf_coeffs[element_id, :]
    hps_error_squared += detBabs * float(
      np.sum(weights * (u_hps - u_true) ** 2)
    )

    c_best = V_eval.T @ (weights * u_true)
    u_best = V_eval @ c_best
    best_error_squared += detBabs * float(
      np.sum(weights * (u_best - u_true) ** 2)
    )
    reference_squared += detBabs * float(np.sum(weights * u_true ** 2))

  denominator = max(reference_squared, 1.0e-300)
  return (
    math.sqrt(hps_error_squared / denominator),
    math.sqrt(best_error_squared / denominator),
  )


def count_mesh_faces(mesh) -> tuple[int, int]:
  boundary = sum(len(owners) == 1 for owners in mesh.face_to_elements.values())
  interior = sum(len(owners) == 2 for owners in mesh.face_to_elements.values())
  return int(boundary), int(interior)


def assert_hps_result(
  result: HpsEllipticResult,
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
  #assert result.leaf_threads_used == 1, result


def check_reduction(
  rows: list[dict[str, float | int | str]],
  min_reduction: float,
) -> None:
  if min_reduction <= 0.0:
    return

  for D in sorted({int(row["D"]) for row in rows}):
    selected = sorted(
      (row for row in rows if int(row["D"]) == D),
      key=lambda row: int(row["n"]),
    )
    if len(selected) < 2:
      continue
    initial = max(
      float(row["relative_L2_error"])
      for row in selected[: min(2, len(selected))]
    )
    final = min(
      float(row["relative_L2_error"])
      for row in selected[-min(2, len(selected)) :]
    )
    reduction = initial / max(final, 1.0e-300)
    assert reduction >= min_reduction, (
      f"D={D}: HPS error reduction {reduction:.3e} "
      f"is below requested {min_reduction:.3e}"
    )


def save_results_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
  fieldnames = [
    "leaf_operator_mode",
    "D", "n", "dimPi", "p", "nverts", "nelem", "M", "m_int", "kf",
    "boundary_faces", "interior_faces", "tree_depth", "metis_splits",
    "fallback_splits", "alpha", "beta", "tau_C", "q_solve_vol",
    "q_solve_face", "q_data_vol", "q_data_face", "q_eval",
    "leaf_threads_used", "root_residual", "interface_residual",
    "parent_residual", "relative_L2_error", "best_relative_L2_error",
    "hps_to_best_L2_ratio",
  ]
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def plot_l2_vs_n(rows: list[dict[str, float | int | str]], path: Path) -> None:
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
  plt.ylabel(r"relative weighted $L^2$ error")
  plt.title(
    "Dense-sparse-leaf HPS elliptic nonpolynomial convergence"
  )
  plt.xticks(sorted({int(row["n"]) for row in rows}))
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def plot_l2_vs_dimPi(rows: list[dict[str, float | int | str]], path: Path) -> None:
  plt.figure(figsize=(7.5, 5.2))
  for D in sorted({int(row["D"]) for row in rows}):
    subset = sorted(
      (row for row in rows if int(row["D"]) == D),
      key=lambda row: int(row["dimPi"]),
    )
    x = np.asarray([int(row["dimPi"]) for row in subset], dtype=np.int64)
    y = np.asarray([float(row["relative_L2_error"]) for row in subset])
    plt.semilogy(x, y, marker="o", label=f"D={D}")

  plt.xlabel(r"$M=\dim(\Pi_n^D)$")
  plt.ylabel(r"relative weighted $L^2$ error")
  plt.title(
    "Dense-sparse-leaf HPS convergence by approximation dimension"
  )
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def plot_hps_vs_best_l2_vs_n(
  rows: list[dict[str, float | int | str]],
  path: Path,
) -> None:
  plt.figure(figsize=(8.4, 5.6))
  for D in sorted({int(row["D"]) for row in rows}):
    subset = sorted(
      (row for row in rows if int(row["D"]) == D),
      key=lambda row: int(row["n"]),
    )
    x = np.asarray([int(row["n"]) for row in subset], dtype=np.int64)
    hps = np.asarray([float(row["relative_L2_error"]) for row in subset])
    best = np.asarray([float(row["best_relative_L2_error"]) for row in subset])

    hps_line, = plt.semilogy(x, hps, marker="o", label=f"HPS D={D}")
    plt.semilogy(
      x,
      best,
      marker="s",
      linestyle="--",
      color=hps_line.get_color(),
      label=f"best local projection D={D}",
    )

  plt.xlabel(r"polynomial degree $n$")
  plt.ylabel(r"relative weighted $L^2$ error")
  plt.title(
    "Dense-sparse-leaf HPS error versus best elementwise projection"
  )
  plt.xticks(sorted({int(row["n"]) for row in rows}))
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend(ncol=2, fontsize="small")
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Nonpolynomial manufactured convergence study for the full "
      "variable-coefficient nondivergence-form elliptic HPS solve, "
      "using only the DenseSparse leaf backend."
    )
  )
  parser.add_argument("--D", default="1,2,3")
  parser.add_argument("--min-n", type=int, default=2)
  parser.add_argument("--max-n", type=int, default=6)
  parser.add_argument("--q-pad", type=int, default=1)
  parser.add_argument(
    "--q-data-factor",
    type=float,
    default=1.5,
    help="data quadrature is max(n+1, round(q_data_factor*n))",
  )
  parser.add_argument(
    "--q-eval-extra",
    type=int,
    default=1,
    help="evaluation quadrature increment above the data quadrature",
  )
  parser.add_argument("--alpha", type=float, default=1.0)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--tau-C", type=float, default=10.0)
  parser.add_argument("--residual-tol", type=float, default=5.0e-10)
  parser.add_argument("--min-reduction", type=float, default=0.0)
  parser.add_argument("--determinant-tol", type=float, default=1.0e-13)
  parser.add_argument("--mesh-seed", type=int, default=61000)
  parser.add_argument("--tree-seed", type=int, default=62000)
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument("--simplify-symbolics", action="store_true")
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  if args.min_n < 2:
    raise ValueError("the elliptic HPS backend requires --min-n >= 2")
  if args.max_n < args.min_n:
    raise ValueError("--max-n must be at least --min-n")
  if args.alpha == 0.0:
    raise ValueError("pure Neumann is not yet supported; use alpha != 0")
  if args.q_pad < 0:
    raise ValueError("--q-pad must be nonnegative")
  if args.q_data_factor < 1.0:
    raise ValueError("--q-data-factor must be at least 1")
  if args.q_eval_extra < 0:
    raise ValueError("--q-eval-extra must be nonnegative")

  # Keep BLAS single-threaded while parallelizing leaf construction with OpenMP.
  # DenseSparse retains only dense L_int at each leaf; trace and flux actions
  # remain CSC, while the explicit HPS Ulam/S maps feed the normal tree passes.
  set_openblas_threads(1)
  set_omp_threads(16)
  print("thread control: OpenBLAS=1, OpenMP=16")
  print(
    "leaf backend: DenseSparse "
    "(dense L_int; CSC T/F; no dense T/F/A_tau; explicit HPS Ulam/S retained)"
  )

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
  rows: list[dict[str, float | int | str]] = []

  for D in dimensions:
    print(f"\nConstructing fixed mesh/tree and symbolic elliptic problem for D={D}")
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
    problem = build_symbolic_problem(D, simplify=args.simplify_symbolics)
    kappa = np.asarray(
      [0.71 + 0.17 * i for i in range(D + 1)],
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
      p = n
      q_solve_vol = n + args.q_pad
      q_solve_face = 1 if D == 1 else n + args.q_pad
      q_data_vol = max(
        n + 1,
        int(math.floor(args.q_data_factor * n + 0.5)),
      )
      q_data_face = 1 if D == 1 else q_data_vol
      q_eval = q_data_vol + args.q_eval_extra

      pc = RefSimplexPrecomp(
        D,
        n,
        kappa,
        q_pad=args.q_pad,
        q_vol=q_solve_vol,
        q_face=q_solve_face,
      )
      pc_data = RefSimplexPrecomp(
        D,
        n,
        kappa,
        q_pad=max(q_data_vol - n, 0),
        q_vol=q_data_vol,
        q_face=q_data_face,
      )

      A_coeffs, b_coeffs, c_coeffs = (
        project_coefficient_fields_elementmajor(
          pc_data,
          vertex_row,
          mesh.X,
          mesh.simplices,
          problem,
          p,
        )
      )
      f_int = project_source_elementmajor(
        pc_data,
        vertex_row,
        mesh.X,
        mesh.simplices,
        problem.f_fun,
      )
      boundary_keys, boundary_g = project_robin_boundary_data(
        pc_data,
        vertex_row,
        mesh.X,
        mesh.simplices,
        mesh.face_to_elements,
        args.alpha,
        args.beta,
        problem.u_fun,
        problem.grad_fun,
      )

      result = run_elliptic_mesh_tree_solve(
        pc,
        vertex_ids,
        mesh.X,
        mesh.simplices,
        tree.merge_pairs,
        A_coeffs,
        b_coeffs,
        c_coeffs,
        f_int,
        boundary_keys,
        boundary_g,
        p2=p,
        p1=p,
        p0=p,
        assume_symmetric=True,
        tau_C=args.tau_C,
        alpha=args.alpha,
        beta=args.beta,
        verbose=args.verbose,
        leaf_operator_mode=HpsLeafOperatorMode.DENSE_SPARSE,
      )
      assert_hps_result(
        result,
        boundary_faces=boundary_faces,
        interior_faces=interior_faces,
        residual_tol=args.residual_tol,
      )

      pc_eval = RefSimplexPrecomp(
        D,
        n,
        kappa,
        q_pad=max(q_eval - n, 0),
        q_vol=q_eval,
        q_face=1 if D == 1 else q_eval,
      )
      relative_l2, best_relative_l2 = global_relative_L2_errors(
        pc_eval,
        vertex_row,
        mesh.X,
        mesh.simplices,
        result.leaf_coeffs,
        problem.u_fun,
      )
      if not np.isfinite(relative_l2) or not np.isfinite(best_relative_l2):
        raise AssertionError(f"non-finite L2 error for D={D}, n={n}")
      hps_to_best = relative_l2 / max(best_relative_l2, 1.0e-300)

      row: dict[str, float | int | str] = {
        "leaf_operator_mode": "DenseSparse",
        "D": D,
        "n": n,
        "dimPi": dimPi(D, n),
        "p": p,
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
        "q_solve_vol": q_solve_vol,
        "q_solve_face": q_solve_face,
        "q_data_vol": q_data_vol,
        "q_data_face": q_data_face,
        "q_eval": q_eval,
        "leaf_threads_used": result.leaf_threads_used,
        "root_residual": result.root_robin_residual_inf,
        "interface_residual": result.interface_flux_residual_inf,
        "parent_residual": result.parent_consistency_residual_inf,
        "relative_L2_error": relative_l2,
        "best_relative_L2_error": best_relative_l2,
        "hps_to_best_L2_ratio": hps_to_best,
      }
      rows.append(row)

      print(
        f"  n={n:2d} dimPi={row['dimPi']:5d} p={p:2d} "
        f"M={result.M:5d} m_int={result.m_int:5d} kf={result.kf:5d} "
        f"rel_L2={relative_l2:.3e} best_L2={best_relative_l2:.3e} "
        f"HPS/best={hps_to_best:.3e} "
        f"root={result.root_robin_residual_inf:.3e} "
        f"iface={result.interface_flux_residual_inf:.3e} "
        f"parent={result.parent_consistency_residual_inf:.3e} "
        f"threads={result.leaf_threads_used}"
      )

  check_reduction(rows, args.min_reduction)

  csv_path = (
    args.output_dir
    / "hps_elliptic_nonpoly_dense_sparse_convergence.csv"
  )
  plot_n_path = (
    args.output_dir
    / "hps_elliptic_nonpoly_dense_sparse_l2_vs_n.png"
  )
  plot_dim_path = (
    args.output_dir
    / "hps_elliptic_nonpoly_dense_sparse_l2_vs_dimPi.png"
  )
  plot_best_path = (
    args.output_dir
    / "hps_elliptic_nonpoly_dense_sparse_hps_vs_best.png"
  )
  save_results_csv(rows, csv_path)
  plot_l2_vs_n(rows, plot_n_path)
  plot_l2_vs_dimPi(rows, plot_dim_path)
  plot_hps_vs_best_l2_vs_n(rows, plot_best_path)

  print(
    "\nall requested DenseSparse-leaf variable-coefficient "
    "elliptic HPS solves completed"
  )
  print(f"results CSV: {csv_path}")
  print(f"L2 versus n plot: {plot_n_path}")
  print(f"L2 versus dimPi plot: {plot_dim_path}")
  print(f"HPS versus best-projection plot: {plot_best_path}")


if __name__ == "__main__":
  main()
