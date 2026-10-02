from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "testing" else _THIS.parent
_PYTHON_DIR = _PROJECT_ROOT / "python"
_TESTING_DIR = _PROJECT_ROOT / "testing"
for _path in (_PYTHON_DIR, _TESTING_DIR):
  if str(_path) not in sys.path:
    sys.path.insert(0, str(_path))

from hps_mesh_tree_driver import (  # noqa: E402
  DEFAULT_QHULL_OPTIONS,
  build_merge_tree,
  generate_mesh,
  validate_merge_tree,
)
from jhps import (  # noqa: E402
  HpsEllipticSolver,
  HpsLeafLeastSquaresSolver,
  HpsLeafOperatorMode,
)
from jprecomp import RefSimplexPrecomp, dimPi  # noqa: E402
from thread_control import set_omp_threads, set_openblas_threads  # noqa: E402
from test_hps_elliptic_mesh_tree_nonpoly_dense_qr_quadrature_tau_base import (  # noqa: E402,E501
  assert_hps_result,
  build_symbolic_problem,
  count_mesh_faces,
  make_vertex_row_map,
  project_coefficient_fields_elementmajor,
  project_robin_boundary_data,
  project_source_elementmajor,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Profile the persistent Dense/DenseQR elliptic HPS path: coefficient "
      "projection, reusable HPS construction, and repeated online solves."
    )
  )
  parser.add_argument("--D", type=int, default=3)
  parser.add_argument("--n", type=int, default=6)
  parser.add_argument(
    "--interior-points",
    type=int,
    default=64,
    help="number of random interior mesh points passed to generate_mesh",
  )
  parser.add_argument("--q-pad", type=int, default=1)
  parser.add_argument("--q-data-factor", type=float, default=1.5)
  parser.add_argument("--alpha", type=float, default=1.0)
  parser.add_argument("--beta", type=float, default=2.3)
  parser.add_argument("--tau-C", dest="tau_C", type=float, default=1.0)
  parser.add_argument("--mesh-seed", type=int, default=61003)
  parser.add_argument("--tree-seed", type=int, default=62003)
  parser.add_argument("--omp-threads", type=int, default=16)
  parser.add_argument("--blas-threads", type=int, default=1)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--reps", type=int, default=25)
  parser.add_argument(
    "--rhs-cases",
    type=int,
    default=4,
    help="number of prebuilt scaled (f,g) pairs cycled through online solves",
  )
  parser.add_argument(
    "--phase",
    choices=("both", "precompute", "solve"),
    default="both",
    help=(
      "both: report construction and online solves; precompute: stop after "
      "persistent construction; solve: repeat online solves enough for profiling"
    ),
  )
  parser.add_argument("--residual-tol", type=float, default=5.0e-8)
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if args.D < 1 or args.D > 5:
    raise ValueError("--D must be in 1..5")
  if args.n < 2:
    raise ValueError("--n must be at least 2")
  if args.interior_points < 0:
    raise ValueError("--interior-points must be nonnegative")
  if args.q_data_factor < 1.0:
    raise ValueError("--q-data-factor must be at least 1")
  if args.omp_threads < 1 or args.blas_threads < 1:
    raise ValueError("thread counts must be positive")
  if args.warmup < 0 or args.reps < 1 or args.rhs_cases < 1:
    raise ValueError("--warmup >= 0, --reps >= 1, and --rhs-cases >= 1 required")
  if args.alpha == 0.0:
    raise ValueError("pure Neumann is not supported by this persistent wrapper")

  set_openblas_threads(args.blas_threads)
  set_omp_threads(args.omp_threads)

  D = args.D
  n = args.n
  p = n
  q_solve_vol = n + args.q_pad
  q_solve_face = 1 if D == 1 else q_solve_vol
  q_data_vol = max(
    n + 1,
    int(math.floor(args.q_data_factor * n + 0.5)),
  )
  q_data_face = 1 if D == 1 else q_data_vol

  print("persistent elliptic HPS profiling driver")
  print(f"  D                    = {D}")
  print(f"  degree n             = {n}")
  print(f"  dimPi                = {dimPi(D, n)}")
  print(f"  requested interior   = {args.interior_points}")
  print(f"  OpenMP threads       = {args.omp_threads}")
  print(f"  OpenBLAS threads     = {args.blas_threads}")
  print(f"  phase                = {args.phase}")

  # --------------------------------------------------------------------------
  # Problem/mesh setup. This is outside the persistent HPS construction time.
  # --------------------------------------------------------------------------
  mesh_rng = np.random.default_rng(args.mesh_seed)
  mesh = generate_mesh(
    D,
    args.interior_points,
    mesh_rng,
    DEFAULT_QHULL_OPTIONS,
    1.0e-13,
  )
  tree = build_merge_tree(
    mesh.adjacency,
    partitioner="pymetis",
    seed=args.tree_seed,
  )
  validate_merge_tree(mesh.adjacency, tree.merge_pairs, tree.root_id)

  vertex_ids = np.arange(mesh.X.shape[0], dtype=np.int32)
  vertex_row = make_vertex_row_map(vertex_ids)
  boundary_faces, interior_faces = count_mesh_faces(mesh)

  print(f"  vertices              = {mesh.X.shape[0]}")
  print(f"  elements/leaves       = {mesh.simplices.shape[0]}")
  print(f"  boundary faces        = {boundary_faces}")
  print(f"  interior faces        = {interior_faces}")
  print(f"  tree depth            = {tree.max_depth}")
  print(f"  merges                = {tree.merge_pairs.shape[0]}")

  problem = build_symbolic_problem(D)
  kappa = np.asarray(
    [0.71 + 0.17 * i for i in range(D + 1)],
    dtype=np.float64,
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
    q_face=q_data_face,
  )

  # --------------------------------------------------------------------------
  # Project variable coefficients and one representative (f,g) pair.
  # This is data preparation, not part of the reusable online HPS solve.
  # --------------------------------------------------------------------------
  t0 = time.perf_counter()

  A_coeffs, b_coeffs, c_coeffs = project_coefficient_fields_elementmajor(
    pc_data,
    vertex_row,
    mesh.X,
    mesh.simplices,
    problem,
    p,
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

  data_projection_time = time.perf_counter() - t0

  print(f"  data projection       = {data_projection_time:.6f} s")

  # Prebuild several linearly scaled cases. Scaling both f and g scales the
  # exact solution of this linear problem, while keeping case preparation out
  # of the timed solve region.
  rhs_cases: list[tuple[np.ndarray, np.ndarray]] = []
  for i in range(args.rhs_cases):
    scale = 1.0 + 0.01 * i
    rhs_cases.append((
      np.ascontiguousarray(scale * f_int),
      np.ascontiguousarray(scale * boundary_g),
    ))

  # --------------------------------------------------------------------------
  # Reusable HPS construction/factorization.
  # --------------------------------------------------------------------------
  t0 = time.perf_counter()

  solver = HpsEllipticSolver(
    pc,
    vertex_ids,
    mesh.X,
    mesh.simplices,
    tree.merge_pairs,
    A_coeffs,
    b_coeffs,
    c_coeffs,
    boundary_keys,
    p2=p,
    p1=p,
    p0=p,
    assume_symmetric=True,
    tau_C=args.tau_C,
    alpha=args.alpha,
    beta=args.beta,
    leaf_operator_mode=HpsLeafOperatorMode.DENSE,
    leaf_least_squares_solver=HpsLeafLeastSquaresSolver.DENSE_QR,
  )

  precompute_time = time.perf_counter() - t0

  print(f"  persistent precompute = {precompute_time:.6f} s")
  print(f"  leaf threads used     = {solver.leaf_threads_used}")
  print(f"  root trace dofs       = {solver.root_nb}")
  print(f"  interface trace dofs  = {solver.interface_nb}")

  if args.phase == "precompute":
    solver.close()
    return

  try:
    # One checked solve validates the persistent path before profiling it.
    checked = solver.solve(
      f_int,
      boundary_g,
      check_residuals=True,
    )
    assert_hps_result(
      checked,
      boundary_faces=boundary_faces,
      interior_faces=interior_faces,
      residual_tol=args.residual_tol,
    )

    print("  checked solve:")
    print(
      f"    root residual       = "
      f"{checked.root_robin_residual_inf:.3e}"
    )
    print(
      f"    interface residual  = "
      f"{checked.interface_flux_residual_inf:.3e}"
    )
    print(
      f"    parent residual     = "
      f"{checked.parent_consistency_residual_inf:.3e}"
    )

    # Warmup: diagnostics off so this follows the online path we want to model.
    for i in range(args.warmup):
      f_case, g_case = rhs_cases[i % len(rhs_cases)]
      solver.solve(f_case, g_case, check_residuals=False)

    # ------------------------------------------------------------------------
    # Repeated reusable solves. Only solver.solve() is timed here.
    # ------------------------------------------------------------------------
    times = np.empty(args.reps, dtype=np.float64)
    checksum = 0.0

    for rep in range(args.reps):
      f_case, g_case = rhs_cases[rep % len(rhs_cases)]

      t0 = time.perf_counter()
      result = solver.solve(
        f_case,
        g_case,
        check_residuals=False,
      )
      times[rep] = time.perf_counter() - t0

      # Consume one value so every returned result is observably used.
      checksum += float(result.leaf_coeffs[0, 0])

    print("online persistent solve timing")
    print(f"  warmup solves         = {args.warmup}")
    print(f"  timed solves          = {args.reps}")
    print(f"  min                   = {np.min(times):.6f} s")
    print(f"  median                = {np.median(times):.6f} s")
    print(f"  mean                  = {np.mean(times):.6f} s")
    print(f"  max                   = {np.max(times):.6f} s")
    print(
      f"  leaves/s (median)     = "
      f"{mesh.simplices.shape[0] / np.median(times):.3e}"
    )
    print(f"  checksum              = {checksum:.16e}")

  finally:
    solver.close()


if __name__ == "__main__":
  main()
