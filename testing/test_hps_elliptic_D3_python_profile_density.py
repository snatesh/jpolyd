from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import test_hps_elliptic_mesh_tree_nonpoly_convergence as base
from jelliptic import EllipticPlan
from jprecomp import RefSimplexPrecomp, dimPi


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "python" else _THIS.parent
_DEFAULT_OUTPUT_DIR = (
  _PROJECT_ROOT / "build" / "hps_elliptic_D3_python_profile_density"
)

# Relative thresholds are applied elementwise as
#
#   |L_ij| > rel_tol * max_ij |L_ij|.
#
# This is preferable to a single absolute tolerance because affine derivative
# scaling changes the magnitude of L from element to element. The 1e-14 and
# 1e-12 levels retain almost everything except roundoff-scale fill, while
# 1e-10 and 1e-8 show increasingly aggressive numerical sparsification.
DEFAULT_RELATIVE_THRESHOLDS = (1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8)


def elapsed_call(fun: Callable[..., Any], *args, **kwargs) -> tuple[Any, float]:
  start = time.perf_counter()
  value = fun(*args, **kwargs)
  return value, time.perf_counter() - start


def parse_thresholds(text: str) -> tuple[float, ...]:
  values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
  if not values:
    raise ValueError("at least one density threshold is required")
  if any(not np.isfinite(value) or value <= 0.0 for value in values):
    raise ValueError("density thresholds must be finite and positive")
  if tuple(sorted(set(values))) != values:
    raise ValueError("density thresholds must be unique and increasing")
  return values


def threshold_tag(value: float) -> str:
  return f"{value:.0e}".replace("+", "").replace("-", "m")


def simplex_face_keys(simplex: np.ndarray) -> frozenset[tuple[int, ...]]:
  simplex = np.asarray(simplex, dtype=np.int64)
  return frozenset(
    tuple(sorted(np.delete(simplex, face_id).astype(int).tolist()))
    for face_id in range(simplex.size)
  )


def merge_structure_records(
  simplices: np.ndarray,
  merge_pairs: np.ndarray,
  kf: int,
) -> list[dict[str, int | float]]:
  nelem = int(simplices.shape[0])
  node_faces: dict[int, frozenset[tuple[int, ...]]] = {
    element_id: simplex_face_keys(simplex)
    for element_id, simplex in enumerate(simplices)
  }
  node_level: dict[int, int] = {element_id: 0 for element_id in range(nelem)}

  records: list[dict[str, int | float]] = []
  for merge_id, pair in enumerate(np.asarray(merge_pairs, dtype=np.int64)):
    child_a = int(pair[0])
    child_b = int(pair[1])
    parent_id = nelem + merge_id

    faces_a = node_faces[child_a]
    faces_b = node_faces[child_b]
    shared = faces_a.intersection(faces_b)
    if not shared:
      raise ValueError(
        f"merge {merge_id} children {child_a}, {child_b} share no boundary face"
      )

    exterior_a = faces_a.difference(shared)
    exterior_b = faces_b.difference(shared)
    parent_faces = exterior_a.union(exterior_b)
    level = max(node_level[child_a], node_level[child_b]) + 1

    n_shared_faces = len(shared)
    nI = n_shared_faces * int(kf)
    nAE = len(exterior_a) * int(kf)
    nBE = len(exterior_b) * int(kf)

    records.append({
      "merge_id": merge_id,
      "parent_id": parent_id,
      "child_a": child_a,
      "child_b": child_b,
      "level": level,
      "shared_faces": n_shared_faces,
      "A_exterior_faces": len(exterior_a),
      "B_exterior_faces": len(exterior_b),
      "parent_boundary_faces": len(parent_faces),
      "nI": nI,
      "nAE": nAE,
      "nBE": nBE,
      "nI_squared": float(nI) ** 2,
      "nI_cubed": float(nI) ** 3,
    })

    node_faces[parent_id] = frozenset(parent_faces)
    node_level[parent_id] = level

  return records


def summarize_merge_structure(
  records: list[dict[str, int | float]],
) -> dict[str, int | float]:
  if not records:
    return {
      "merge_count": 0,
      "merge_max_level": 0,
      "merge_max_shared_faces": 0,
      "merge_max_nI": 0,
      "merge_mean_nI": 0.0,
      "merge_sum_nI2": 0.0,
      "merge_sum_nI3": 0.0,
      "merge_lu_flop_proxy": 0.0,
    }

  nI = np.asarray([float(record["nI"]) for record in records])
  return {
    "merge_count": len(records),
    "merge_max_level": max(int(record["level"]) for record in records),
    "merge_max_shared_faces": max(
      int(record["shared_faces"]) for record in records
    ),
    "merge_max_nI": int(np.max(nI)),
    "merge_mean_nI": float(np.mean(nI)),
    "merge_sum_nI2": float(np.sum(nI ** 2)),
    "merge_sum_nI3": float(np.sum(nI ** 3)),
    "merge_lu_flop_proxy": float((2.0 / 3.0) * np.sum(nI ** 3)),
  }


def mesh_quality_summary(
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
) -> dict[str, float]:
  det_values = []
  cond_values = []
  sigma_min_values = []
  sigma_max_values = []

  for simplex in simplices:
    V_phys = base.element_vertices(vertex_row, coords, simplex)
    B = V_phys[:, 1:] - V_phys[:, [0]]
    singular_values = np.linalg.svd(B, compute_uv=False)
    det_values.append(abs(float(np.linalg.det(B))))
    sigma_min_values.append(float(singular_values[-1]))
    sigma_max_values.append(float(singular_values[0]))
    cond_values.append(float(singular_values[0] / singular_values[-1]))

  det = np.asarray(det_values)
  cond = np.asarray(cond_values)
  sigma_min = np.asarray(sigma_min_values)
  sigma_max = np.asarray(sigma_max_values)

  return {
    "mesh_detB_min": float(np.min(det)),
    "mesh_detB_median": float(np.median(det)),
    "mesh_detB_max": float(np.max(det)),
    "mesh_cond2_min": float(np.min(cond)),
    "mesh_cond2_median": float(np.median(cond)),
    "mesh_cond2_max": float(np.max(cond)),
    "mesh_sigma_min_min": float(np.min(sigma_min)),
    "mesh_sigma_min_median": float(np.median(sigma_min)),
    "mesh_sigma_max_median": float(np.median(sigma_max)),
    "mesh_sigma_max_max": float(np.max(sigma_max)),
  }


def csc_storage_bytes(nnz: int, ncols: int) -> int:
  # Estimated conventional CSC storage with float64 values and int32 indices.
  return int(nnz) * (8 + 4) + (int(ncols) + 1) * 4


def audit_element_L_density(
  pc: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  A_coeffs: np.ndarray,
  b_coeffs: np.ndarray,
  c_coeffs: np.ndarray,
  p: int,
  relative_thresholds: tuple[float, ...],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
  plan_start = time.perf_counter()
  plan = EllipticPlan(
    pc,
    p2=p,
    p1=p,
    p0=p,
    assume_symmetric=True,
  )
  workspace = plan.create_workspace()
  plan_seconds = time.perf_counter() - plan_start

  element_records: list[dict[str, float | int]] = []
  total_entries = 0
  exact_nnz_total = 0
  threshold_nnz_totals = {threshold: 0 for threshold in relative_thresholds}
  threshold_csc_bytes = {threshold: 0 for threshold in relative_thresholds}
  density_by_threshold = {threshold: [] for threshold in relative_thresholds}
  max_abs_values = []
  nonzero_relative_values: list[np.ndarray] = []
  assembly_start = time.perf_counter()

  try:
    out = np.empty((plan.m2, plan.M), dtype=np.float64, order="F")

    for element_id, simplex in enumerate(simplices):
      V_phys = base.element_vertices(vertex_row, coords, simplex)
      B = V_phys[:, 1:] - V_phys[:, [0]]
      detBabs = abs(float(np.linalg.det(B)))
      BinvT = np.asfortranarray(np.linalg.inv(B).T)

      L = plan.assemble_L_int(
        BinvT,
        detBabs,
        A=A_coeffs[element_id],
        b=b_coeffs[element_id],
        c=c_coeffs[element_id],
        workspace=workspace,
        out=out,
      )

      abs_L = np.abs(L)
      max_abs = float(np.max(abs_L)) if abs_L.size else 0.0
      entries = int(abs_L.size)
      exact_nnz = int(np.count_nonzero(abs_L))
      total_entries += entries
      exact_nnz_total += exact_nnz
      max_abs_values.append(max_abs)

      if max_abs > 0.0:
        positive = abs_L[abs_L > 0.0] / max_abs
        if positive.size:
          nonzero_relative_values.append(np.asarray(positive, dtype=np.float64))

      base_record: dict[str, float | int] = {
        "element_id": element_id,
        "detBabs": detBabs,
        "max_abs_L": max_abs,
        "entries": entries,
        "exact_nnz": exact_nnz,
        "exact_density": exact_nnz / max(entries, 1),
        "dense_bytes": entries * 8,
      }

      for threshold in relative_thresholds:
        cutoff = threshold * max_abs
        nnz = int(np.count_nonzero(abs_L > cutoff)) if max_abs > 0.0 else 0
        density = nnz / max(entries, 1)
        csc_bytes = csc_storage_bytes(nnz, plan.M)
        tag = threshold_tag(threshold)

        base_record[f"cutoff_rel_{tag}"] = cutoff
        base_record[f"nnz_rel_{tag}"] = nnz
        base_record[f"density_rel_{tag}"] = density
        base_record[f"csc_bytes_rel_{tag}"] = csc_bytes
        base_record[f"dense_to_csc_rel_{tag}"] = (
          (entries * 8) / max(csc_bytes, 1)
        )

        threshold_nnz_totals[threshold] += nnz
        threshold_csc_bytes[threshold] += csc_bytes
        density_by_threshold[threshold].append(density)

      element_records.append(base_record)
  finally:
    assembly_seconds = time.perf_counter() - assembly_start
    workspace.close()
    plan.close()

  dense_bytes_total = total_entries * 8
  summary: dict[str, float | int] = {
    "L_plan_seconds": plan_seconds,
    "L_assembly_seconds": assembly_seconds,
    "L_audit_seconds": plan_seconds + assembly_seconds,
    "L_rows": int(pc.m_int),
    "L_cols": int(pc.M),
    "L_entries_per_element": int(pc.m_int * pc.M),
    "L_total_entries": total_entries,
    "L_exact_nnz_total": exact_nnz_total,
    "L_exact_density_total": exact_nnz_total / max(total_entries, 1),
    "L_dense_bytes_total": dense_bytes_total,
    "L_max_abs_min": float(np.min(max_abs_values)),
    "L_max_abs_median": float(np.median(max_abs_values)),
    "L_max_abs_max": float(np.max(max_abs_values)),
  }

  if nonzero_relative_values:
    relative_magnitudes = np.concatenate(nonzero_relative_values)
    summary.update({
      "L_relmag_min_positive": float(np.min(relative_magnitudes)),
      "L_relmag_q001": float(np.quantile(relative_magnitudes, 0.001)),
      "L_relmag_q01": float(np.quantile(relative_magnitudes, 0.01)),
      "L_relmag_q10": float(np.quantile(relative_magnitudes, 0.10)),
      "L_relmag_median": float(np.median(relative_magnitudes)),
    })
  else:
    summary.update({
      "L_relmag_min_positive": 0.0,
      "L_relmag_q001": 0.0,
      "L_relmag_q01": 0.0,
      "L_relmag_q10": 0.0,
      "L_relmag_median": 0.0,
    })

  for threshold in relative_thresholds:
    tag = threshold_tag(threshold)
    nnz_total = threshold_nnz_totals[threshold]
    csc_bytes_total = threshold_csc_bytes[threshold]
    densities = np.asarray(density_by_threshold[threshold])
    summary.update({
      f"L_nnz_total_rel_{tag}": nnz_total,
      f"L_density_total_rel_{tag}": nnz_total / max(total_entries, 1),
      f"L_density_min_rel_{tag}": float(np.min(densities)),
      f"L_density_mean_rel_{tag}": float(np.mean(densities)),
      f"L_density_max_rel_{tag}": float(np.max(densities)),
      f"L_csc_bytes_total_rel_{tag}": csc_bytes_total,
      f"L_dense_to_csc_rel_{tag}": (
        dense_bytes_total / max(csc_bytes_total, 1)
      ),
    })

  return summary, element_records


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames: list[str] = []
  seen = set()
  for row in rows:
    for key in row:
      if key not in seen:
        fieldnames.append(key)
        seen.add(key)

  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def plot_timings(rows: list[dict[str, Any]], path: Path) -> None:
  phases = [
    ("precomp_solve_seconds", "solve precompute"),
    ("precomp_data_seconds", "data precompute"),
    ("coefficient_projection_seconds", "coefficient projection"),
    ("source_projection_seconds", "source projection"),
    ("boundary_projection_seconds", "boundary projection"),
    ("L_audit_seconds", "extra L audit"),
    ("hps_solve_seconds", "opaque C++ HPS solve"),
    ("error_evaluation_seconds", "error evaluation"),
  ]

  x = np.asarray([int(row["n"]) for row in rows])
  plt.figure(figsize=(9.0, 5.8))
  for key, label in phases:
    y = np.asarray([max(float(row[key]), 1.0e-12) for row in rows])
    plt.semilogy(x, y, marker="o", label=label)

  plt.xlabel(r"polynomial degree $n$")
  plt.ylabel("wall time (seconds)")
  plt.title("D=3 elliptic HPS Python-visible phase timings")
  plt.xticks(x)
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend(fontsize="small", ncol=2)
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def plot_L_density(
  rows: list[dict[str, Any]],
  relative_thresholds: tuple[float, ...],
  path: Path,
) -> None:
  x = np.asarray([int(row["n"]) for row in rows])
  plt.figure(figsize=(8.2, 5.5))

  for threshold in relative_thresholds:
    tag = threshold_tag(threshold)
    y = np.asarray([
      float(row[f"L_density_total_rel_{tag}"])
      for row in rows
    ])
    plt.plot(x, y, marker="o", label=rf"$|L_{{ij}}|>{threshold:.0e}\,\max|L|$")

  plt.xlabel(r"polynomial degree $n$")
  plt.ylabel("aggregate numerical density of $L$")
  plt.title("D=3 assembled elliptic leaf PDE-block density")
  plt.xticks(x)
  plt.ylim(bottom=0.0, top=1.02)
  plt.grid(True, linewidth=0.5)
  plt.legend(fontsize="small")
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def plot_error(rows: list[dict[str, Any]], path: Path) -> None:
  x = np.asarray([int(row["n"]) for row in rows])
  hps = np.asarray([float(row["relative_L2_error"]) for row in rows])
  best = np.asarray([float(row["best_relative_L2_error"]) for row in rows])

  plt.figure(figsize=(7.6, 5.2))
  plt.semilogy(x, hps, marker="o", label="HPS")
  plt.semilogy(x, best, marker="s", linestyle="--", label="best local projection")
  plt.xlabel(r"polynomial degree $n$")
  plt.ylabel(r"relative weighted $L^2$ error")
  plt.title("D=3 elliptic HPS verification during density audit")
  plt.xticks(x)
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Python-only D=3 timing, topology, mesh-quality, and assembled-L "
      "density audit layered on the full nonpolynomial elliptic HPS test."
    )
  )
  parser.add_argument("--min-n", type=int, default=2)
  parser.add_argument("--max-n", type=int, default=6)
  parser.add_argument("--q-pad", type=int, default=1)
  parser.add_argument("--q-data-factor", type=float, default=1.5)
  parser.add_argument("--q-eval-extra", type=int, default=1)
  parser.add_argument("--alpha", type=float, default=1.0)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--tau-C", type=float, default=10.0)
  parser.add_argument("--residual-tol", type=float, default=5.0e-10)
  parser.add_argument("--determinant-tol", type=float, default=1.0e-13)
  parser.add_argument("--mesh-seed", type=int, default=61000)
  parser.add_argument("--tree-seed", type=int, default=62000)
  parser.add_argument(
    "--interior-points",
    type=int,
    default=int(base.DEFAULT_INTERIOR_POINT_COUNTS[3]),
    help="number of random interior points added to the D=3 cube corners",
  )
  parser.add_argument(
    "--density-thresholds",
    default=",".join(f"{value:.0e}" for value in DEFAULT_RELATIVE_THRESHOLDS),
    help=(
      "increasing elementwise relative thresholds, applied as "
      "tol*max(abs(L)); default: 1e-14,1e-12,1e-10,1e-8"
    ),
  )
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument("--simplify-symbolics", action="store_true")
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  D = 3
  if args.min_n < 2:
    raise ValueError("use --min-n >= 2 for the second-order elliptic operator")
  if args.max_n < args.min_n:
    raise ValueError("--max-n must be at least --min-n")
  if args.q_pad < 0:
    raise ValueError("--q-pad must be nonnegative")
  if args.q_data_factor < 1.0:
    raise ValueError("--q-data-factor must be at least 1")
  if args.q_eval_extra < 0:
    raise ValueError("--q-eval-extra must be nonnegative")
  if args.alpha == 0.0:
    raise ValueError("pure Neumann is not implemented")
  if args.interior_points < 0:
    raise ValueError("--interior-points must be nonnegative")

  relative_thresholds = parse_thresholds(args.density_thresholds)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  # The current Fortran LSMR shim is serial. Keep both inner runtimes serial.
  base.set_openblas_threads(1)
  base.set_omp_threads(1)
  print("thread control: OpenBLAS=1, OpenMP=1")
  print(
    "L numerical-zero thresholds: "
    + ", ".join(f"{value:.0e} * max(abs(L_e))" for value in relative_thresholds)
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
    f"module={getattr(pymetis, '__file__', '<unknown>')}"
  )

  setup_start = time.perf_counter()
  mesh_rng = np.random.default_rng(args.mesh_seed + D)
  mesh, mesh_seconds = elapsed_call(
    base.generate_mesh,
    D,
    args.interior_points,
    mesh_rng,
    base.DEFAULT_QHULL_OPTIONS,
    args.determinant_tol,
  )
  tree, tree_seconds = elapsed_call(
    base.build_merge_tree,
    mesh.adjacency,
    partitioner="pymetis",
    seed=args.tree_seed + D,
  )
  base.validate_merge_tree(mesh.adjacency, tree.merge_pairs, tree.root_id)
  problem, symbolic_seconds = elapsed_call(
    base.build_symbolic_problem,
    D,
    simplify=args.simplify_symbolics,
  )
  setup_seconds = time.perf_counter() - setup_start

  vertex_ids = np.arange(mesh.X.shape[0], dtype=np.int32)
  vertex_row = base.make_vertex_row_map(vertex_ids)
  boundary_faces, interior_faces = base.count_mesh_faces(mesh)
  quality = mesh_quality_summary(vertex_row, mesh.X, mesh.simplices)
  kappa = np.asarray(
    [0.71 + 0.17 * i for i in range(D + 1)],
    dtype=np.float64,
  )

  mesh_file = base.export_case(args.output_dir, mesh, tree)
  figure_file = base.visualize_case(args.output_dir, mesh, tree, show=False)
  print(
    f"D=3: interior_points={args.interior_points} "
    f"nverts={mesh.X.shape[0]} nelem={mesh.simplices.shape[0]} "
    f"boundary_faces={boundary_faces} interior_faces={interior_faces} "
    f"merges={tree.merge_pairs.shape[0]} depth={tree.max_depth}"
  )
  print(
    f"mesh quality: detB min/median/max="
    f"{quality['mesh_detB_min']:.3e}/"
    f"{quality['mesh_detB_median']:.3e}/"
    f"{quality['mesh_detB_max']:.3e}; "
    f"cond2 min/median/max="
    f"{quality['mesh_cond2_min']:.3e}/"
    f"{quality['mesh_cond2_median']:.3e}/"
    f"{quality['mesh_cond2_max']:.3e}"
  )
  print(f"mesh/tree data: {mesh_file}")
  if figure_file is not None:
    print(f"visualization: {figure_file}")

  summary_rows: list[dict[str, Any]] = []
  element_density_rows: list[dict[str, Any]] = []
  merge_rows: list[dict[str, Any]] = []

  for n in range(args.min_n, args.max_n + 1):
    degree_start = time.perf_counter()
    p = n
    q_solve_vol = n + args.q_pad
    q_solve_face = n + args.q_pad
    q_data_vol = max(
      n + 1,
      int(math.floor(args.q_data_factor * n + 0.5)),
    )
    q_data_face = q_data_vol
    q_eval = q_data_vol + args.q_eval_extra

    pc, precomp_solve_seconds = elapsed_call(
      RefSimplexPrecomp,
      D,
      n,
      kappa,
      q_pad=args.q_pad,
      q_vol=q_solve_vol,
      q_face=q_solve_face,
    )
    pc_data, precomp_data_seconds = elapsed_call(
      RefSimplexPrecomp,
      D,
      n,
      kappa,
      q_pad=max(q_data_vol - n, 0),
      q_vol=q_data_vol,
      q_face=q_data_face,
    )

    coeffs, coefficient_projection_seconds = elapsed_call(
      base.project_coefficient_fields_elementmajor,
      pc_data,
      vertex_row,
      mesh.X,
      mesh.simplices,
      problem,
      p,
    )
    A_coeffs, b_coeffs, c_coeffs = coeffs

    f_int, source_projection_seconds = elapsed_call(
      base.project_source_elementmajor,
      pc_data,
      vertex_row,
      mesh.X,
      mesh.simplices,
      problem.f_fun,
    )
    boundary_data, boundary_projection_seconds = elapsed_call(
      base.project_robin_boundary_data,
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
    boundary_keys, boundary_g = boundary_data

    L_summary, L_records = audit_element_L_density(
      pc,
      vertex_row,
      mesh.X,
      mesh.simplices,
      A_coeffs,
      b_coeffs,
      c_coeffs,
      p,
      relative_thresholds,
    )
    for record in L_records:
      element_density_rows.append({"D": D, "n": n, "p": p, **record})

    solve_start = time.perf_counter()
    result = base.run_elliptic_mesh_tree_solve(
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
    )
    hps_solve_seconds = time.perf_counter() - solve_start

    base.assert_hps_result(
      result,
      boundary_faces=boundary_faces,
      interior_faces=interior_faces,
      residual_tol=args.residual_tol,
    )

    pc_eval, precomp_eval_seconds = elapsed_call(
      RefSimplexPrecomp,
      D,
      n,
      kappa,
      q_pad=max(q_eval - n, 0),
      q_vol=q_eval,
      q_face=q_eval,
    )
    errors, error_evaluation_seconds = elapsed_call(
      base.global_relative_L2_errors,
      pc_eval,
      vertex_row,
      mesh.X,
      mesh.simplices,
      result.leaf_coeffs,
      problem.u_fun,
    )
    relative_l2, best_relative_l2 = errors

    merge_records = merge_structure_records(
      mesh.simplices,
      tree.merge_pairs,
      result.kf,
    )
    merge_summary = summarize_merge_structure(merge_records)
    for record in merge_records:
      merge_rows.append({"D": D, "n": n, "kf": result.kf, **record})

    nb = (D + 1) * result.kf
    leaf_rows = result.m_int + nb
    leaf_dense_entries = leaf_rows * result.M
    total_leaf_lsmr_solves = result.nelem * (nb + 1)
    leaf_dense_scan_proxy = total_leaf_lsmr_solves * leaf_dense_entries
    degree_total_seconds = time.perf_counter() - degree_start

    row: dict[str, Any] = {
      "D": D,
      "n": n,
      "p": p,
      "dimPi": dimPi(D, n),
      "nverts": int(mesh.X.shape[0]),
      "nelem": int(mesh.simplices.shape[0]),
      "M": result.M,
      "m_int": result.m_int,
      "kf": result.kf,
      "nb": nb,
      "leaf_rows": leaf_rows,
      "leaf_cols": result.M,
      "leaf_dense_entries_per_element": leaf_dense_entries,
      "leaf_trace_rhs_per_element": nb,
      "total_leaf_lsmr_solves_proxy": total_leaf_lsmr_solves,
      "leaf_dense_scan_proxy_per_iteration": leaf_dense_scan_proxy,
      "boundary_faces": boundary_faces,
      "interior_faces": interior_faces,
      "tree_depth": tree.max_depth,
      "q_solve_vol": q_solve_vol,
      "q_solve_face": q_solve_face,
      "q_data_vol": q_data_vol,
      "q_data_face": q_data_face,
      "q_eval": q_eval,
      "mesh_generation_seconds": mesh_seconds,
      "tree_build_seconds": tree_seconds,
      "symbolic_build_seconds": symbolic_seconds,
      "setup_seconds": setup_seconds,
      "precomp_solve_seconds": precomp_solve_seconds,
      "precomp_data_seconds": precomp_data_seconds,
      "coefficient_projection_seconds": coefficient_projection_seconds,
      "source_projection_seconds": source_projection_seconds,
      "boundary_projection_seconds": boundary_projection_seconds,
      "hps_solve_seconds": hps_solve_seconds,
      "precomp_eval_seconds": precomp_eval_seconds,
      "error_evaluation_seconds": error_evaluation_seconds,
      "degree_total_seconds": degree_total_seconds,
      "root_residual": result.root_robin_residual_inf,
      "interface_residual": result.interface_flux_residual_inf,
      "parent_residual": result.parent_consistency_residual_inf,
      "relative_L2_error": relative_l2,
      "best_relative_L2_error": best_relative_l2,
      "hps_to_best_L2_ratio": relative_l2 / max(best_relative_l2, 1.0e-300),
      "leaf_threads_used": result.leaf_threads_used,
      **quality,
      **merge_summary,
      **L_summary,
    }
    summary_rows.append(row)

    density_text = " ".join(
      f"rho[{threshold:.0e}]="
      f"{row[f'L_density_total_rel_{threshold_tag(threshold)}']:.3f}"
      for threshold in relative_thresholds
    )
    compression_text = " ".join(
      f"dense/CSC[{threshold:.0e}]="
      f"{row[f'L_dense_to_csc_rel_{threshold_tag(threshold)}']:.2f}x"
      for threshold in relative_thresholds
    )
    print(
      f"n={n:2d} M={result.M:4d} m2={result.m_int:4d} "
      f"kf={result.kf:4d} nb={nb:4d} nelem={result.nelem:3d} "
      f"HPS={hps_solve_seconds:.3f}s L_audit={L_summary['L_audit_seconds']:.3f}s "
      f"rel_L2={relative_l2:.3e} best={best_relative_l2:.3e}"
    )
    print(f"  {density_text}")
    print(f"  {compression_text}")
    print(
      f"  L relative-magnitude quantiles: min={L_summary['L_relmag_min_positive']:.3e} "
      f"q0.1%={L_summary['L_relmag_q001']:.3e} "
      f"q1%={L_summary['L_relmag_q01']:.3e} "
      f"q10%={L_summary['L_relmag_q10']:.3e}"
    )
    print(
      f"  merge max nI={merge_summary['merge_max_nI']} "
      f"sum(nI^3)={merge_summary['merge_sum_nI3']:.3e}; "
      f"leaf solve-count proxy={total_leaf_lsmr_solves}"
    )

  summary_csv = args.output_dir / "D3_profile_density_summary.csv"
  element_csv = args.output_dir / "D3_L_density_by_element.csv"
  merge_csv = args.output_dir / "D3_merge_structure_by_degree.csv"
  timing_plot = args.output_dir / "D3_phase_timings_vs_n.png"
  density_plot = args.output_dir / "D3_L_density_vs_n.png"
  error_plot = args.output_dir / "D3_verification_error_vs_n.png"

  save_csv(summary_csv, summary_rows)
  save_csv(element_csv, element_density_rows)
  save_csv(merge_csv, merge_rows)
  plot_timings(summary_rows, timing_plot)
  plot_L_density(summary_rows, relative_thresholds, density_plot)
  plot_error(summary_rows, error_plot)

  print("\nD=3 Python-only elliptic HPS profile and density audit completed")
  print(f"summary CSV: {summary_csv}")
  print(f"element density CSV: {element_csv}")
  print(f"merge structure CSV: {merge_csv}")
  print(f"timing plot: {timing_plot}")
  print(f"L density plot: {density_plot}")
  print(f"verification plot: {error_plot}")


if __name__ == "__main__":
  main()
