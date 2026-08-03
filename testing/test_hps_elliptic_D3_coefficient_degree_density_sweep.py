from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import test_hps_elliptic_mesh_tree_nonpoly_convergence as base
from jelliptic import EllipticPlan
from jprecomp import RefSimplexPrecomp, dimPi
from thread_control import set_omp_threads, set_openblas_threads


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "python" else _THIS.parent
_DEFAULT_OUTPUT_DIR = (
  _PROJECT_ROOT / "build" / "hps_elliptic_D3_coefficient_degree_density"
)

# Applied elementwise:
#
#   |L_ij| > rel_tol * max_kl |L_kl|.
#
# The smallest levels distinguish exact/structural zeros from roundoff fill;
# 1e-10 and 1e-8 estimate increasingly aggressive numerical sparsification.
DEFAULT_RELATIVE_THRESHOLDS = (1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8)


def parse_int_list(text: str, name: str) -> tuple[int, ...]:
  values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
  if not values:
    raise ValueError(f"{name} must contain at least one integer")
  if len(set(values)) != len(values):
    raise ValueError(f"{name} must not contain duplicate values")
  return values


def parse_thresholds(text: str) -> tuple[float, ...]:
  values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
  if not values:
    raise ValueError("--thresholds must contain at least one value")
  if any(not np.isfinite(value) or value <= 0.0 for value in values):
    raise ValueError("all thresholds must be finite and positive")
  if len(set(values)) != len(values):
    raise ValueError("--thresholds must not contain duplicate values")
  return tuple(sorted(values))


def threshold_tag(value: float) -> str:
  return f"{value:.0e}".replace("+", "").replace("-", "m")


def csc_storage_bytes(nnz: int, ncols: int) -> int:
  # Conventional CSC estimate: float64 values, int32 row indices and colptr.
  return int(nnz) * (8 + 4) + (int(ncols) + 1) * 4


def project_coefficients_at_degree(
  pc_data: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  problem: base.SymbolicProblem,
  p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  start = time.perf_counter()
  values = base.project_coefficient_fields_elementmajor(
    pc_data,
    vertex_row,
    coords,
    simplices,
    problem,
    p,
  )
  return (*values, time.perf_counter() - start)


def audit_L_for_degree(
  pc: RefSimplexPrecomp,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  A_coeffs: np.ndarray,
  b_coeffs: np.ndarray,
  c_coeffs: np.ndarray,
  p: int,
  thresholds: tuple[float, ...],
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

  element_rows: list[dict[str, float | int]] = []
  total_entries = 0
  exact_nnz_total = 0
  threshold_nnz_total = {threshold: 0 for threshold in thresholds}
  threshold_densities = {threshold: [] for threshold in thresholds}
  threshold_csc_total = {threshold: 0 for threshold in thresholds}
  positive_relative_entries: list[np.ndarray] = []
  max_abs_values: list[float] = []

  assembly_start = time.perf_counter()
  try:
    out = np.empty((plan.mR, plan.M), dtype=np.float64, order="F")

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
      entries = int(abs_L.size)
      exact_nnz = int(np.count_nonzero(abs_L))
      max_abs = float(np.max(abs_L)) if entries else 0.0

      total_entries += entries
      exact_nnz_total += exact_nnz
      max_abs_values.append(max_abs)

      if max_abs > 0.0:
        positive = abs_L[abs_L > 0.0] / max_abs
        if positive.size:
          positive_relative_entries.append(np.asarray(positive).copy())

      row: dict[str, float | int] = {
        "p": p,
        "element_id": element_id,
        "rows": int(L.shape[0]),
        "cols": int(L.shape[1]),
        "entries": entries,
        "exact_nnz": exact_nnz,
        "exact_density": exact_nnz / max(entries, 1),
        "max_abs_L": max_abs,
        "detBabs": detBabs,
        "cond2_B": float(np.linalg.cond(B)),
      }

      for threshold in thresholds:
        if max_abs == 0.0:
          nnz = 0
        else:
          nnz = int(np.count_nonzero(abs_L > threshold * max_abs))
        density = nnz / max(entries, 1)
        tag = threshold_tag(threshold)

        threshold_nnz_total[threshold] += nnz
        threshold_densities[threshold].append(density)
        threshold_csc_total[threshold] += csc_storage_bytes(nnz, L.shape[1])

        row[f"nnz_rel_{tag}"] = nnz
        row[f"density_rel_{tag}"] = density

      element_rows.append(row)
  finally:
    assembly_seconds = time.perf_counter() - assembly_start
    workspace.close()
    plan.close()

  dense_bytes = total_entries * 8
  if positive_relative_entries:
    relative_entries = np.concatenate(positive_relative_entries)
    quantiles = np.quantile(
      relative_entries,
      [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0],
    )
  else:
    quantiles = np.zeros(7, dtype=np.float64)

  summary: dict[str, float | int] = {
    "p": p,
    "Mp": dimPi(pc.D, p),
    "L_rows": int(plan.mR),
    "L_cols": int(plan.M),
    "nelem": int(simplices.shape[0]),
    "total_entries": total_entries,
    "exact_nnz_total": exact_nnz_total,
    "exact_density_aggregate": exact_nnz_total / max(total_entries, 1),
    "dense_bytes_total": dense_bytes,
    "plan_seconds": plan_seconds,
    "assembly_seconds": assembly_seconds,
    "max_abs_L_min": float(np.min(max_abs_values)),
    "max_abs_L_mean": float(np.mean(max_abs_values)),
    "max_abs_L_max": float(np.max(max_abs_values)),
    "relative_abs_q0000": float(quantiles[0]),
    "relative_abs_q0001": float(quantiles[1]),
    "relative_abs_q0010": float(quantiles[2]),
    "relative_abs_q0100": float(quantiles[3]),
    "relative_abs_q0500": float(quantiles[4]),
    "relative_abs_q0900": float(quantiles[5]),
    "relative_abs_q1000": float(quantiles[6]),
    # These are the full Clenshaw function-space degrees used by jelliptic.
    "K2": pc.n - 2 + p,
    "MK2": dimPi(pc.D, pc.n - 2 + p),
    "K1": pc.n - 1 + p,
    "MK1": dimPi(pc.D, pc.n - 1 + p),
    "K0": pc.n + p,
    "MK0": dimPi(pc.D, pc.n + p),
  }

  for threshold in thresholds:
    tag = threshold_tag(threshold)
    nnz = threshold_nnz_total[threshold]
    csc_bytes = threshold_csc_total[threshold]
    densities = np.asarray(threshold_densities[threshold], dtype=np.float64)

    summary[f"nnz_rel_{tag}_total"] = nnz
    summary[f"density_rel_{tag}_aggregate"] = nnz / max(total_entries, 1)
    summary[f"density_rel_{tag}_min"] = float(np.min(densities))
    summary[f"density_rel_{tag}_mean"] = float(np.mean(densities))
    summary[f"density_rel_{tag}_max"] = float(np.max(densities))
    summary[f"csc_bytes_rel_{tag}_total"] = csc_bytes
    summary[f"dense_to_csc_rel_{tag}"] = (
      dense_bytes / csc_bytes if csc_bytes > 0 else math.inf
    )

  return summary, element_rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys())
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def plot_density(
  summaries: list[dict[str, float | int]],
  thresholds: tuple[float, ...],
  path: Path,
) -> None:
  plt.figure(figsize=(8.2, 5.5))
  p_values = np.asarray([int(row["p"]) for row in summaries], dtype=np.int64)

  for threshold in thresholds:
    tag = threshold_tag(threshold)
    density = np.asarray([
      float(row[f"density_rel_{tag}_aggregate"])
      for row in summaries
    ])
    plt.plot(
      p_values,
      density,
      marker="o",
      label=rf"$|L_{{ij}}|>{threshold:.0e}\max|L|$",
    )

  plt.xlabel(r"coefficient degree $p=p_2=p_1=p_0$")
  plt.ylabel(r"aggregate numerical density of $L$")
  plt.title(
    rf"$D=3$, fixed solution degree $n={int(summaries[0]['n'])}$: "
    "elliptic leaf PDE-block density"
  )
  plt.xticks(p_values)
  plt.ylim(bottom=0.0, top=1.03)
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def plot_storage_ratio(
  summaries: list[dict[str, float | int]],
  thresholds: tuple[float, ...],
  path: Path,
) -> None:
  plt.figure(figsize=(8.2, 5.5))
  p_values = np.asarray([int(row["p"]) for row in summaries], dtype=np.int64)

  for threshold in thresholds:
    tag = threshold_tag(threshold)
    ratio = np.asarray([
      float(row[f"dense_to_csc_rel_{tag}"])
      for row in summaries
    ])
    plt.plot(
      p_values,
      ratio,
      marker="o",
      label=rf"drop threshold {threshold:.0e}",
    )

  plt.axhline(1.0, linestyle="--", linewidth=1.0, label="dense = CSC")
  plt.xlabel(r"coefficient degree $p=p_2=p_1=p_0$")
  plt.ylabel("dense bytes / estimated CSC bytes")
  plt.title(
    rf"$D=3$, fixed solution degree $n={int(summaries[0]['n'])}$: "
    "dense versus CSC storage"
  )
  plt.xticks(p_values)
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def plot_timing(
  summaries: list[dict[str, float | int]],
  path: Path,
) -> None:
  plt.figure(figsize=(8.2, 5.5))
  p_values = np.asarray([int(row["p"]) for row in summaries], dtype=np.int64)
  projection = np.asarray([
    float(row["coefficient_projection_seconds"]) for row in summaries
  ])
  plan = np.asarray([float(row["plan_seconds"]) for row in summaries])
  assembly = np.asarray([float(row["assembly_seconds"]) for row in summaries])

  plt.semilogy(p_values, projection, marker="o", label="coefficient projection")
  plt.semilogy(p_values, plan, marker="o", label="EllipticPlan construction")
  plt.semilogy(p_values, assembly, marker="o", label="assemble all element L")

  plt.xlabel(r"coefficient degree $p=p_2=p_1=p_0$")
  plt.ylabel("wall time (seconds)")
  plt.title(
    rf"$D=3$, fixed solution degree $n={int(summaries[0]['n'])}$: "
    "coefficient-degree audit timings"
  )
  plt.xticks(p_values)
  plt.grid(True, which="both", linewidth=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig(path, dpi=220)
  plt.close()


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Sweep the common coefficient degree p at fixed D=3 and solution degree "
      "n, assemble every elliptic leaf PDE block L, and measure numerical "
      "density and estimated dense-versus-CSC storage."
    )
  )
  parser.add_argument("--n", type=int, default=6)
  parser.add_argument("--degrees", default="0,1,2,3,4,6")
  parser.add_argument(
    "--thresholds",
    default=",".join(f"{value:.0e}" for value in DEFAULT_RELATIVE_THRESHOLDS),
  )
  parser.add_argument(
    "--interior-points",
    type=int,
    default=int(base.DEFAULT_INTERIOR_POINT_COUNTS[3]),
    help="number of random interior points added to the unit cube",
  )
  parser.add_argument("--q-pad", type=int, default=1)
  parser.add_argument(
    "--q-data-factor",
    type=float,
    default=1.5,
    help="data quadrature is max(n+1, round(q_data_factor*n))",
  )
  parser.add_argument("--mesh-seed", type=int, default=61003)
  parser.add_argument("--determinant-tol", type=float, default=1.0e-13)
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument("--simplify-symbolics", action="store_true")
  args = parser.parse_args()

  D = 3
  n = int(args.n)
  if n < 2:
    raise ValueError("--n must be at least 2")
  if args.interior_points < 0:
    raise ValueError("--interior-points must be nonnegative")
  if args.q_pad < 0:
    raise ValueError("--q-pad must be nonnegative")
  if args.q_data_factor < 1.0:
    raise ValueError("--q-data-factor must be at least 1")

  degrees = parse_int_list(args.degrees, "--degrees")
  if any(p < 0 or p > n for p in degrees):
    raise ValueError("every coefficient degree must satisfy 0 <= p <= n")
  degrees = tuple(sorted(degrees))
  thresholds = parse_thresholds(args.thresholds)

  set_openblas_threads(1)
  set_omp_threads(1)
  print("thread control: OpenBLAS=1, OpenMP=1")

  args.output_dir.mkdir(parents=True, exist_ok=True)

  mesh_rng = np.random.default_rng(args.mesh_seed)
  mesh = base.generate_mesh(
    D,
    args.interior_points,
    mesh_rng,
    base.DEFAULT_QHULL_OPTIONS,
    args.determinant_tol,
  )
  vertex_ids = np.arange(mesh.X.shape[0], dtype=np.int32)
  vertex_row = base.make_vertex_row_map(vertex_ids)
  problem = base.build_symbolic_problem(
    D,
    simplify=args.simplify_symbolics,
  )
  kappa = np.asarray(
    [0.71 + 0.17 * i for i in range(D + 1)],
    dtype=np.float64,
  )

  q_solve_vol = n + args.q_pad
  q_solve_face = n + args.q_pad
  q_data_vol = max(
    n + 1,
    int(math.floor(args.q_data_factor * n + 0.5)),
  )

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
    q_face=q_data_vol,
  )

  print(
    f"D=3 n={n} nverts={mesh.X.shape[0]} "
    f"nelem={mesh.simplices.shape[0]} interior_points={args.interior_points}"
  )
  print(
    f"M={pc.M} m2={pc.m_int} kf={pc.kf} "
    f"thresholds={thresholds}"
  )

  summaries: list[dict[str, float | int]] = []
  element_rows: list[dict[str, float | int]] = []

  for p in degrees:
    A_coeffs, b_coeffs, c_coeffs, projection_seconds = (
      project_coefficients_at_degree(
        pc_data,
        vertex_row,
        mesh.X,
        mesh.simplices,
        problem,
        p,
      )
    )

    summary, rows = audit_L_for_degree(
      pc,
      vertex_row,
      mesh.X,
      mesh.simplices,
      A_coeffs,
      b_coeffs,
      c_coeffs,
      p,
      thresholds,
    )
    summary["D"] = D
    summary["n"] = n
    summary["interior_points"] = int(args.interior_points)
    summary["nverts"] = int(mesh.X.shape[0])
    summary["coefficient_projection_seconds"] = projection_seconds
    summary["q_solve_vol"] = q_solve_vol
    summary["q_data_vol"] = q_data_vol

    # Keep stable leading columns in the CSV.
    ordered = {
      "D": summary.pop("D"),
      "n": summary.pop("n"),
      "p": summary["p"],
      "Mp": summary["Mp"],
      "interior_points": summary.pop("interior_points"),
      "nverts": summary.pop("nverts"),
      "nelem": summary["nelem"],
      "L_rows": summary["L_rows"],
      "L_cols": summary["L_cols"],
      "q_solve_vol": summary.pop("q_solve_vol"),
      "q_data_vol": summary.pop("q_data_vol"),
      "coefficient_projection_seconds": summary.pop(
        "coefficient_projection_seconds"
      ),
    }
    for key, value in summary.items():
      if key not in ordered:
        ordered[key] = value

    summaries.append(ordered)
    element_rows.extend(rows)

    density_text = " ".join(
      f"rho({threshold:.0e})="
      f"{ordered[f'density_rel_{threshold_tag(threshold)}_aggregate']:.4f}"
      for threshold in thresholds
    )
    storage_text = " ".join(
      f"dense/CSC({threshold:.0e})="
      f"{ordered[f'dense_to_csc_rel_{threshold_tag(threshold)}']:.3f}"
      for threshold in thresholds
    )
    print(
      f"p={p:2d} Mp={dimPi(D, p):4d} "
      f"K2/MK2={ordered['K2']}/{ordered['MK2']} "
      f"K1/MK1={ordered['K1']}/{ordered['MK1']} "
      f"K0/MK0={ordered['K0']}/{ordered['MK0']} "
      f"plan={ordered['plan_seconds']:.3e}s "
      f"assembly={ordered['assembly_seconds']:.3e}s"
    )
    print(f"  {density_text}")
    print(f"  {storage_text}")

  summary_path = args.output_dir / "D3_coefficient_degree_density_summary.csv"
  element_path = args.output_dir / "D3_coefficient_degree_density_by_element.csv"
  density_plot = args.output_dir / "D3_L_density_vs_coefficient_degree.png"
  storage_plot = args.output_dir / "D3_dense_vs_CSC_vs_coefficient_degree.png"
  timing_plot = args.output_dir / "D3_timing_vs_coefficient_degree.png"

  write_csv(summary_path, summaries)
  write_csv(element_path, element_rows)
  plot_density(summaries, thresholds, density_plot)
  plot_storage_ratio(summaries, thresholds, storage_plot)
  plot_timing(summaries, timing_plot)

  print("\ncoefficient-degree density sweep completed")
  print(f"summary CSV: {summary_path}")
  print(f"element CSV: {element_path}")
  print(f"density plot: {density_plot}")
  print(f"storage plot: {storage_plot}")
  print(f"timing plot: {timing_plot}")


if __name__ == "__main__":
  main()
