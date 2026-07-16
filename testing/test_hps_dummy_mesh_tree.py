from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np

from jhps import HpsDummyResult, run_mesh_tree_test
from hps_mesh_tree_driver import (
  DEFAULT_QHULL_OPTIONS,
  build_merge_tree,
  export_case,
  generate_mesh,
  validate_merge_tree,
  visualize_case,
)


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1] if _THIS.parent.name == "python" else _THIS.parent
_OUTPUT_DIR = _PROJECT_ROOT / "build" / "hps_dummy_mesh_tree_test"

# Keep D=4,5 modest because the dummy verification forms and solves a dense
# monolithic skeleton system. The D=5 cube corners alone still generate a
# nontrivial mesh in the current SciPy/Qhull triangulation.
INTERIOR_POINT_COUNTS = {
  1: 8,
  2: 4,
  3: 2,
  4: 1,
  5: 0,
}


def count_mesh_faces(mesh) -> tuple[int, int]:
  nboundary = sum(
    1 for owners in mesh.face_to_elements.values() if len(owners) == 1
  )
  ninterior = sum(
    1 for owners in mesh.face_to_elements.values() if len(owners) == 2
  )
  return nboundary, ninterior


def assert_dummy_result(
  result: HpsDummyResult,
  *,
  nboundary_faces: int,
  ninterior_faces: int,
  kf: int,
  tol: float,
) -> None:
  residuals = np.asarray([
    result.root_robin_residual_inf,
    result.interface_flux_residual_inf,
    result.parent_consistency_residual_inf,
    result.monolithic_trace_residual_inf,
  ])

  assert np.all(np.isfinite(residuals)), result
  assert result.root_robin_residual_inf < tol, result
  assert result.interface_flux_residual_inf < tol, result
  assert result.parent_consistency_residual_inf < tol, result
  assert result.monolithic_trace_residual_inf < tol, result
  assert np.isfinite(result.leaf_volume_norm_inf), result
  assert result.leaf_volume_norm_inf > 0.0, result

  # Every root boundary face contributes one block of kf trace unknowns.
  assert result.root_nb == nboundary_faces * kf, result

  # Every interior mesh face is eliminated exactly once somewhere in the
  # externally supplied binary tree, even if one merge interface has many faces.
  assert result.interface_nb == ninterior_faces * kf, result


def main() -> None:
  tol = 5.0e-9
  kf = 1
  alpha = 1.0
  beta = 1.0
  determinant_tol = 1.0e-13

  _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  try:
    import pymetis
  except ImportError as exc:
    raise RuntimeError(
      "This test requires PyMetis. Run it with the same Python interpreter "
      "in which `import pymetis` succeeds."
    ) from exc

  try:
    pymetis_version = importlib.metadata.version("pymetis")
  except importlib.metadata.PackageNotFoundError:
    pymetis_version = "unknown"
  pymetis_path = getattr(pymetis, "__file__", "<unknown>")
  partitioner = "pymetis"
  print(
    "mesh-tree partitioner: pymetis "
    f"version={pymetis_version} module={pymetis_path}; "
    "validated fallback enabled"
  )

  results: list[HpsDummyResult] = []
  total_metis_splits = 0
  total_fallback_splits = 0

  for D in range(1, 6):
    mesh_rng = np.random.default_rng(10000 + D)
    mesh = generate_mesh(
      D,
      INTERIOR_POINT_COUNTS[D],
      mesh_rng,
      DEFAULT_QHULL_OPTIONS,
      determinant_tol,
    )

    tree = build_merge_tree(
      mesh.adjacency,
      partitioner=partitioner,
      seed=20000 + D,
    )
    validate_merge_tree(mesh.adjacency, tree.merge_pairs, tree.root_id)
    assert tree.metis_splits + tree.fallback_splits == mesh.simplices.shape[0] - 1, tree
    assert tree.metis_splits > 0, tree

    # Retain generated inputs for reproducing a failing backend case and save
    # the D=1,2,3 mesh/tree visualizations without opening a GUI.
    mesh_file = export_case(_OUTPUT_DIR, mesh, tree)
    figure_file = visualize_case(_OUTPUT_DIR, mesh, tree, show=False)

    vertex_ids = np.arange(mesh.X.shape[0], dtype=np.int32)
    result = run_mesh_tree_test(
      D,
      vertex_ids,
      mesh.X,
      mesh.simplices,
      tree.merge_pairs,
      kf=kf,
      vol_dim=D + 3,
      seed=30000 + D,
      alpha=alpha,
      beta=beta,
      verbose=True,
    )

    nboundary_faces, ninterior_faces = count_mesh_faces(mesh)
    assert_dummy_result(
      result,
      nboundary_faces=nboundary_faces,
      ninterior_faces=ninterior_faces,
      kf=kf,
      tol=tol,
    )

    total_metis_splits += tree.metis_splits
    total_fallback_splits += tree.fallback_splits
    results.append(result)

    print(
      f"D={D}: nverts={mesh.X.shape[0]} nelem={mesh.simplices.shape[0]} "
      f"boundary_faces={nboundary_faces} interior_faces={ninterior_faces} "
      f"merges={tree.merge_pairs.shape[0]} depth={tree.max_depth} "
      f"metis={tree.metis_splits} fallback={tree.fallback_splits}"
    )
    print(f"  mesh/tree data: {mesh_file}")
    if figure_file is not None:
      print(f"  visualization: {figure_file}")
    print(f"  result: {result}")

  assert total_metis_splits > 0

  print("\nall D=1..5 external SciPy/PyMetis mesh-tree dummy HPS tests passed")
  for result in results:
    print(result)


if __name__ == "__main__":
  main()
