from __future__ import annotations

import argparse
import math
import sys
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
from jprecomp import RefSimplexPrecomp, dimPi  # noqa: E402
from test_hps_elliptic_mesh_tree_nonpoly_dense_qr_quadrature_tau_base import (  # noqa: E402,E501
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
      "Export one representative variable-coefficient elliptic HPS case "
      "to raw binary files for the native C++ profiling driver."
    )
  )
  parser.add_argument("--D", type=int, default=3)
  parser.add_argument("--n", type=int, default=6)
  parser.add_argument("--interior-points", type=int, default=64)
  parser.add_argument("--q-pad", type=int, default=1)
  parser.add_argument("--q-data-factor", type=float, default=1.5)
  parser.add_argument("--alpha", type=float, default=1.0)
  parser.add_argument("--beta", type=float, default=2.3)
  parser.add_argument("--tau-C", dest="tau_C", type=float, default=1.0)
  parser.add_argument("--mesh-seed", type=int, default=61003)
  parser.add_argument("--tree-seed", type=int, default=62003)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=_PROJECT_ROOT / "build" / "hps_profile_case",
  )
  return parser.parse_args()


def write_f64(path: Path, array: np.ndarray) -> None:
  data = np.ascontiguousarray(array, dtype="<f8")
  data.tofile(path)


def write_i32(path: Path, array: np.ndarray) -> None:
  data = np.ascontiguousarray(array, dtype="<i4")
  data.tofile(path)


def main() -> None:
  args = parse_args()

  if args.D < 1 or args.D > 5:
    raise ValueError("--D must be in 1..5")
  if args.n < 2:
    raise ValueError("--n must be at least 2")
  if args.interior_points < 0:
    raise ValueError("--interior-points must be nonnegative")
  if args.q_pad < 0:
    raise ValueError("--q-pad must be nonnegative")
  if args.q_data_factor < 1.0:
    raise ValueError("--q-data-factor must be at least 1")
  if args.alpha == 0.0:
    raise ValueError("pure Neumann is not supported by the current HPS wrapper")
  if args.tau_C <= 0.0:
    raise ValueError("--tau-C must be positive")

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

  problem = build_symbolic_problem(D)
  kappa = np.asarray(
    [0.71 + 0.17 * i for i in range(D + 1)],
    dtype=np.float64,
  )

  pc_data = RefSimplexPrecomp(
    D,
    n,
    kappa,
    q_pad=max(q_data_vol - n, 0),
    q_vol=q_data_vol,
    q_face=q_data_face,
  )

  print("building representative HPS profiling case")
  print(f"  D                    = {D}")
  print(f"  degree n             = {n}")
  print(f"  dimPi                = {dimPi(D, n)}")
  print(f"  requested interior   = {args.interior_points}")
  print(f"  vertices             = {mesh.X.shape[0]}")
  print(f"  elements/leaves      = {mesh.simplices.shape[0]}")
  print(f"  boundary faces       = {boundary_faces}")
  print(f"  interior faces       = {interior_faces}")
  print(f"  tree depth           = {tree.max_depth}")
  print(f"  merges               = {tree.merge_pairs.shape[0]}")

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

  out = args.output_dir
  out.mkdir(parents=True, exist_ok=True)

  # Keep the format intentionally simple: one text metadata file plus raw
  # little-endian arrays. The C++ profiling executable has no file-format
  # dependency beyond the standard library.
  write_f64(out / "kappa.f64", kappa)
  write_i32(out / "vertex_ids.i32", vertex_ids)
  write_f64(out / "coords.f64", mesh.X)
  write_i32(out / "simplices.i32", mesh.simplices)
  write_i32(out / "merge_pairs.i32", tree.merge_pairs)
  write_i32(out / "boundary_keys.i32", boundary_keys)
  write_f64(out / "A_coeffs.f64", A_coeffs)
  write_f64(out / "b_coeffs.f64", b_coeffs)
  write_f64(out / "c_coeffs.f64", c_coeffs)
  write_f64(out / "f_int.f64", f_int)
  write_f64(out / "boundary_g.f64", boundary_g)

  metadata = {
    "format_version": 1,
    "D": D,
    "n": n,
    "q_pad": args.q_pad,
    "q_vol": q_solve_vol,
    "q_face": q_solve_face,
    "p2": p,
    "p1": p,
    "p0": p,
    "assume_symmetric": 1,
    "nverts": int(vertex_ids.size),
    "nelem": int(mesh.simplices.shape[0]),
    "nmerge": int(tree.merge_pairs.shape[0]),
    "nboundary_faces": int(boundary_keys.shape[0]),
    "tau_C": float(args.tau_C),
    "alpha": float(args.alpha),
    "beta": float(args.beta),
    "boundary_faces": int(boundary_faces),
    "interior_faces": int(interior_faces),
    "tree_depth": int(tree.max_depth),
    "A_count": int(A_coeffs.size),
    "b_count": int(b_coeffs.size),
    "c_count": int(c_coeffs.size),
    "f_count": int(f_int.size),
    "g_count": int(boundary_g.size),
  }

  with (out / "meta.txt").open("w", encoding="utf-8") as stream:
    for key, value in metadata.items():
      stream.write(f"{key} {value}\n")

  print("\nexported native profiling case")
  print(f"  directory            = {out}")
  print(f"  A coefficients       = {A_coeffs.shape}")
  print(f"  b coefficients       = {b_coeffs.shape}")
  print(f"  c coefficients       = {c_coeffs.shape}")
  print(f"  source               = {f_int.shape}")
  print(f"  boundary data        = {boundary_g.shape}")
  print("\nnext:")
  print(
    f"  bin/profile_hps_elliptic_native {out} "
    "--warmup 2 --reps 300 --rhs-cases 4"
  )


if __name__ == "__main__":
  main()
