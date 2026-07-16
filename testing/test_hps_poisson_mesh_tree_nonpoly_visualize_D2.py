from __future__ import annotations

import argparse
import importlib.metadata
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jhps import run_poisson_mesh_tree_solve
from hps_mesh_tree_driver import (
  DEFAULT_QHULL_OPTIONS,
  build_merge_tree,
  export_case,
  generate_mesh,
  validate_merge_tree,
  visualize_case,
)
from jbasis import jbasis_build_structures, jbasis_eval_all
from jprecomp import RefSimplexPrecomp
from test_hps_poisson_mesh_tree_nonpoly_convergence import (
  affine_map_ref_to_phys,
  assert_hps_result,
  count_mesh_faces,
  element_vertices,
  eval_scalar_D,
  global_relative_L2_error,
  make_nonpoly_u_f_grad_D,
  make_vertex_row_map,
  project_robin_boundary_data,
  project_source_elementmajor,
)


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "python" else _THIS.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "build" / "hps_poisson_nonpoly_visualize_D2"


def reference_triangle_lattice(resolution: int) -> np.ndarray:
  """Return a uniform lattice on the reference triangle.

  The reference simplex is

    x >= 0, y >= 0, x + y <= 1.

  ``resolution`` is the number of equal subdivisions along each edge.
  """
  resolution = int(resolution)
  if resolution < 2:
    raise ValueError("plot resolution must be at least 2")

  points = [
    (i / resolution, j / resolution)
    for i in range(resolution + 1)
    for j in range(resolution + 1 - i)
  ]
  return np.ascontiguousarray(points, dtype=np.float64)

def evaluate_leaf_solutions_on_lattice(
  n: int,
  kappa: np.ndarray,
  vertex_row: dict[int, int],
  coords: np.ndarray,
  simplices: np.ndarray,
  leaf_coeffs: np.ndarray,
  u_fun,
  resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Evaluate exact and HPS solutions at identical physical sample points."""
  D = 2
  Xhat = reference_triangle_lattice(resolution)

  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)
  V_plot = jbasis_eval_all(
    Xhat,
    kappa,
    n,
    alpha_table,
    tail_deg,
    inv_h,
    D,
  )

  point_blocks: list[np.ndarray] = []
  exact_blocks: list[np.ndarray] = []
  computed_blocks: list[np.ndarray] = []

  for element_id, simplex in enumerate(simplices):
    V_phys = element_vertices(vertex_row, coords, simplex)
    points = affine_map_ref_to_phys(V_phys, Xhat)
    exact_values = eval_scalar_D(u_fun, points)
    computed_values = V_plot @ leaf_coeffs[element_id, :]

    point_blocks.append(points)
    exact_blocks.append(exact_values)
    computed_blocks.append(computed_values)

  return (
    np.ascontiguousarray(np.vstack(point_blocks), dtype=np.float64),
    np.ascontiguousarray(np.concatenate(exact_blocks), dtype=np.float64),
    np.ascontiguousarray(np.concatenate(computed_blocks), dtype=np.float64),
  )

def set_common_3d_axes(
  ax,
  *,
  zmin: float,
  zmax: float,
  elevation: float,
  azimuth: float,
) -> None:
  span = max(zmax - zmin, 1.0e-14)
  zpad = 0.04 * span

  ax.set_xlabel(r"$x_0$")
  ax.set_ylabel(r"$x_1$")
  ax.set_zlabel(r"$u$")
  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(0.0, 1.0)
  ax.set_zlim(zmin - zpad, zmax + zpad)
  ax.view_init(elev=elevation, azim=azimuth)
  ax.set_box_aspect((1.0, 1.0, 0.65))

def save_solution_figure(
  points: np.ndarray,
  exact_values: np.ndarray,
  computed_values: np.ndarray,
  *,
  D: int,
  n: int,
  alpha: float,
  beta: float,
  tau_C: float,
  path: Path,
  elevation: float,
  azimuth: float,
  exact_marker_stride: int,
) -> None:
  """Show and then save one interactive exact-versus-HPS scatter plot."""
  zmin = float(min(np.min(exact_values), np.min(computed_values)))
  zmax = float(max(np.max(exact_values), np.max(computed_values)))

  exact_marker_stride = int(exact_marker_stride)
  if exact_marker_stride < 1:
    raise ValueError("exact_marker_stride must be positive")
  exact_indices = np.arange(0, points.shape[0], exact_marker_stride)

  cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
  computed_color = cycle_colors[0] if cycle_colors else None
  exact_color = cycle_colors[1] if len(cycle_colors) > 1 else computed_color

  fig = plt.figure(figsize=(9.2, 7.2))
  ax = fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)

  # Draw the smaller filled HPS circles first. The larger hollow exact circles
  # are then drawn on top at the identical (x0, x1) sample locations.
  ax.scatter(
    points[:, 0],
    points[:, 1],
    computed_values,
    marker="o",
    s=12,
    color=computed_color,
    depthshade=False,
    zorder=5,
    label="HPS approximation",
  )
  ax.scatter(
    points[exact_indices, 0],
    points[exact_indices, 1],
    exact_values[exact_indices],
    marker="o",
    facecolors="none",
    edgecolors=exact_color,
    linewidths=1.15,
    s=44,
    depthshade=False,
    zorder=10,
    label="Exact solution",
  )

  set_common_3d_axes(
    ax,
    zmin=zmin,
    zmax=zmax,
    elevation=elevation,
    azimuth=azimuth,
  )
  ax.set_title(
    rf"Exact and HPS values at identical points, $D={D}$, $n={n}$"
    "\n"
    rf"$\alpha={alpha:g}$, $\beta={beta:g}$, $\tau_C={tau_C:g}$"
  )
  ax.legend(loc="best")
  fig.tight_layout()

  # Show the interactive window first. Saving occurs only after the window is
  # closed, so the final view angle chosen interactively is preserved.
  plt.show()
  fig.savefig(path, dpi=220, bbox_inches="tight")
  plt.close(fig)

def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Solve the D=2 nonpolynomial manufactured Poisson/Robin problem once "
      "and visualize the reconstructed HPS solution against direct exact "
      "function evaluations at identical points."
    )
  )
  parser.add_argument("--n", type=int, default=8)
  parser.add_argument("--q-pad", type=int, default=1)
  parser.add_argument("--q-data-pad", type=int, default=8)
  parser.add_argument("--q-eval-pad", type=int, default=8)
  parser.add_argument("--alpha", type=float, default=1.0)
  parser.add_argument("--beta", type=float, default=2.3)
  parser.add_argument("--tau-C", type=float, default=0.5)
  parser.add_argument("--interior-points", type=int, default=1)
  parser.add_argument("--plot-resolution", type=int, default=18)
  parser.add_argument(
    "--exact-marker-stride",
    type=int,
    default=1,
    help=(
      "Plot one exact-value ring for every this many lattice samples. "
      "Use 1 to draw every exact sample."
    ),
  )
  parser.add_argument("--mesh-seed", type=int, default=51002)
  parser.add_argument("--tree-seed", type=int, default=52002)
  parser.add_argument("--determinant-tol", type=float, default=1.0e-13)
  parser.add_argument("--residual-tol", type=float, default=5.0e-8)
  parser.add_argument("--elevation", type=float, default=28.0)
  parser.add_argument("--azimuth", type=float, default=-125.0)
  parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
  parser.add_argument("--simplify-symbolics", action="store_true")
  parser.add_argument("--verbose", action="store_true")
  args = parser.parse_args()

  D = 2
  if args.n < 2:
    raise ValueError("the current C++ Poisson HPS backend requires n >= 2")
  if args.alpha == 0.0:
    raise ValueError("pure Neumann is not yet supported; use alpha != 0")
  if args.tau_C <= 0.0 or not np.isfinite(args.tau_C):
    raise ValueError("--tau-C must be finite and positive")
  if args.interior_points < 0:
    raise ValueError("--interior-points must be nonnegative")
  if args.exact_marker_stride < 1:
    raise ValueError("--exact-marker-stride must be positive")
  if min(args.q_pad, args.q_data_pad, args.q_eval_pad) < 0:
    raise ValueError("quadrature padding values must be nonnegative")

  try:
    import pymetis
  except ImportError as exc:
    raise RuntimeError("this visualization requires PyMetis") from exc
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

  mesh_rng = np.random.default_rng(args.mesh_seed)
  mesh = generate_mesh(
    D,
    args.interior_points,
    mesh_rng,
    DEFAULT_QHULL_OPTIONS,
    args.determinant_tol,
  )
  tree = build_merge_tree(
    mesh.adjacency,
    partitioner="pymetis",
    seed=args.tree_seed,
  )
  validate_merge_tree(mesh.adjacency, tree.merge_pairs, tree.root_id)

  mesh_file = export_case(args.output_dir, mesh, tree)
  mesh_figure = visualize_case(args.output_dir, mesh, tree, show=False)

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

  q_solve = args.n + args.q_pad
  pc_solve = RefSimplexPrecomp(
    D,
    args.n,
    kappa,
    q_pad=args.q_pad,
    q_vol=q_solve,
    q_face=q_solve,
  )

  q_data = args.n + args.q_data_pad
  pc_data = RefSimplexPrecomp(
    D,
    args.n,
    kappa,
    q_pad=args.q_data_pad,
    q_vol=q_data,
    q_face=q_data,
  )

  f_int = project_source_elementmajor(
    pc_data,
    vertex_row,
    mesh.X,
    mesh.simplices,
    f_fun,
  )
  boundary_keys, boundary_g = project_robin_boundary_data(
    pc_data,
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
    pc_solve,
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

  q_eval = args.n + args.q_eval_pad
  pc_eval = RefSimplexPrecomp(
    D,
    args.n,
    kappa,
    q_pad=args.q_eval_pad,
    q_vol=q_eval,
    q_face=q_eval,
  )
  relative_l2 = global_relative_L2_error(
    pc_eval,
    vertex_row,
    mesh.X,
    mesh.simplices,
    result.leaf_coeffs,
    u_fun,
  )

  points, exact_values, computed_values = (
    evaluate_leaf_solutions_on_lattice(
      args.n,
      kappa,
      vertex_row,
      mesh.X,
      mesh.simplices,
      result.leaf_coeffs,
      u_fun,
      args.plot_resolution,
    )
  )

  max_point_error = float(np.max(np.abs(computed_values - exact_values)))
  rms_point_error = math.sqrt(float(np.mean((computed_values - exact_values) ** 2)))

  figure_path = (
    args.output_dir
    / f"hps_poisson_nonpoly_solution_D2_n{args.n}.png"
  )
  data_path = (
    args.output_dir
    / f"hps_poisson_nonpoly_solution_D2_n{args.n}.npz"
  )

  save_solution_figure(
    points,
    exact_values,
    computed_values,
    D=D,
    n=args.n,
    alpha=args.alpha,
    beta=args.beta,
    tau_C=args.tau_C,
    path=figure_path,
    elevation=args.elevation,
    azimuth=args.azimuth,
    exact_marker_stride=args.exact_marker_stride,
  )

  np.savez_compressed(
    data_path,
    X=mesh.X,
    simplices=mesh.simplices,
    merge_pairs=tree.merge_pairs,
    leaf_coeffs=result.leaf_coeffs,
    sample_points=points,
    exact_values=exact_values,
    computed_values=computed_values,
    n=np.asarray(args.n, dtype=np.int32),
    kappa=kappa,
    alpha=np.asarray(args.alpha, dtype=np.float64),
    beta=np.asarray(args.beta, dtype=np.float64),
    tau_C=np.asarray(args.tau_C, dtype=np.float64),
    relative_weighted_L2_error=np.asarray(relative_l2, dtype=np.float64),
  )

  print(
    f"D=2 n={args.n} nverts={mesh.X.shape[0]} "
    f"nelem={mesh.simplices.shape[0]} M={result.M} "
    f"m_int={result.m_int} kf={result.kf}"
  )
  print(
    f"  quadrature: solve={q_solve} data={q_data} eval={q_eval}"
  )
  print(
    f"  residual infs: root={result.root_robin_residual_inf:.3e} "
    f"iface={result.interface_flux_residual_inf:.3e} "
    f"parent={result.parent_consistency_residual_inf:.3e}"
  )
  print(f"  global relative weighted L2 error = {relative_l2:.6e}")
  print(f"  sampled max point error          = {max_point_error:.6e}")
  print(f"  sampled RMS point error          = {rms_point_error:.6e}")
  print(f"  mesh/tree data: {mesh_file}")
  if mesh_figure is not None:
    print(f"  mesh/tree visualization: {mesh_figure}")
  print(f"  solution figure: {figure_path}")
  print(f"  sampled solution data: {data_path}")


if __name__ == "__main__":
  main()
