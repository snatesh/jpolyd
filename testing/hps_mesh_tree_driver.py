#!/usr/bin/env python3
"""Generate simplex meshes and HPS merge trees for D=1,...,5.

The script owns the Python side of the mesh/tree pipeline:

  point cloud -> simplicial mesh -> codimension-one face incidence
  -> simplex dual graph -> recursive binary merge tree -> validation/export/viz

It runs without the C dummy HPS backend. When a shared library is supplied with
``--lib``, it also calls the generic C entry point
``jhps_dummy_mesh_tree_test`` through ctypes.

Required packages:
  numpy
  scipy
  matplotlib       (only for D=1,2,3 visualization)

Partitioning package:
  pymetis          (required by the default partitioner)

Example:
  python hps_mesh_tree_driver.py
  python hps_mesh_tree_driver.py --partitioner fallback --show
  python hps_mesh_tree_driver.py --lib ./libjhps_dummy.so --degree 4
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
Adjacency = list[set[int]]


class PyMetisPartitionError(RuntimeError):
  """PyMetis ran, but did not return a usable connected bisection."""



DEFAULT_INTERIOR_COUNTS = {
  1: 24,
  2: 40,
  3: 24,
  4: 12,
  5: 8,
}

DEFAULT_QHULL_OPTIONS = "Qbb Qc Qz Qx Q12"


@dataclass(frozen=True)
class MeshData:
  D: int
  X: FloatArray
  simplices: IntArray
  face_to_elements: dict[tuple[int, ...], tuple[int, ...]]
  adjacency: Adjacency
  simplex_determinants: FloatArray


@dataclass(frozen=True)
class TreeData:
  merge_pairs: IntArray
  root_id: int
  leaf_order: IntArray
  metis_splits: int
  fallback_splits: int
  max_depth: int
  mean_leaf_depth: float
  max_child_imbalance: float
  max_interface_faces: int


@dataclass(frozen=True)
class BackendResult:
  return_code: int
  root_robin_residual_inf: float
  interface_flux_residual_inf: float
  parent_consistency_residual_inf: float
  monolithic_trace_residual_inf: float
  leaf_volume_norm_inf: float
  root_nb: int
  nfaces: int


@dataclass(frozen=True)
class DimensionSummary:
  D: int
  nverts: int
  nelem: int
  nfaces: int
  nboundary_faces: int
  ninterior_faces: int
  ndual_edges: int
  min_abs_simplex_determinant: float
  max_abs_simplex_determinant: float
  root_id: int
  nmerge: int
  tree_max_depth: int
  tree_mean_leaf_depth: float
  tree_max_child_imbalance: float
  tree_max_interface_faces: int
  metis_splits: int
  fallback_splits: int
  mesh_file: str
  figure_file: str | None
  backend: BackendResult | None


def parse_count_map(text: str) -> dict[int, int]:
  """Parse a string such as ``1:24,2:40,3:24,4:12,5:8``."""
  result: dict[int, int] = {}
  if not text.strip():
    return result

  for item in text.split(","):
    fields = item.strip().split(":")
    if len(fields) != 2:
      raise argparse.ArgumentTypeError(
        f"invalid interior-count entry {item!r}; expected D:N"
      )
    try:
      D = int(fields[0])
      count = int(fields[1])
    except ValueError as exc:
      raise argparse.ArgumentTypeError(
        f"invalid interior-count entry {item!r}; D and N must be integers"
      ) from exc
    if D < 1:
      raise argparse.ArgumentTypeError("dimensions must be positive")
    if count < 0:
      raise argparse.ArgumentTypeError("interior point counts must be nonnegative")
    result[D] = count

  return result


def cube_corners(D: int) -> FloatArray:
  return np.asarray(list(product((0.0, 1.0), repeat=D)), dtype=np.float64)


def orient_simplices_positive(
  X: FloatArray,
  simplices: IntArray,
  determinant_tol: float,
) -> tuple[IntArray, FloatArray]:
  """Orient every D-simplex positively and reject degenerate simplices."""
  oriented = np.ascontiguousarray(simplices, dtype=np.int32).copy()
  edge_matrices = X[oriented[:, 1:]] - X[oriented[:, [0]]]
  determinants = np.linalg.det(edge_matrices)

  bad = np.flatnonzero(np.abs(determinants) <= determinant_tol)
  if bad.size:
    preview = ", ".join(str(int(i)) for i in bad[:10])
    raise ValueError(
      f"found {bad.size} simplex/simplices with |det| <= {determinant_tol:g}; "
      f"first indices: {preview}"
    )

  negative = determinants < 0.0
  if np.any(negative):
    tmp = oriented[negative, 0].copy()
    oriented[negative, 0] = oriented[negative, 1]
    oriented[negative, 1] = tmp
    determinants[negative] *= -1.0

  return oriented, np.ascontiguousarray(determinants, dtype=np.float64)


def generate_mesh(
  D: int,
  n_interior: int,
  rng: np.random.Generator,
  qhull_options: str,
  determinant_tol: float,
) -> MeshData:
  """Generate a conforming simplex mesh of the unit D-cube."""
  if D < 1:
    raise ValueError("D must be at least 1")

  if D == 1:
    interior = rng.random(n_interior, dtype=np.float64)
    coordinates = np.concatenate((np.array([0.0, 1.0]), interior))
    X = np.sort(coordinates).reshape(-1, 1)
    simplices = np.column_stack(
      (np.arange(X.shape[0] - 1), np.arange(1, X.shape[0]))
    ).astype(np.int32)
  else:
    corners = cube_corners(D)
    interior = rng.random((n_interior, D), dtype=np.float64)
    X = np.ascontiguousarray(np.vstack((corners, interior)), dtype=np.float64)

    try:
      triangulation = Delaunay(X, qhull_options=qhull_options)
    except QhullError as exc:
      raise RuntimeError(
        f"Delaunay failed in D={D} with qhull_options={qhull_options!r}"
      ) from exc

    simplices = np.ascontiguousarray(triangulation.simplices, dtype=np.int32)

    used = np.unique(simplices)
    unused = np.setdiff1d(np.arange(X.shape[0]), used, assume_unique=True)
    if unused.size:
      preview = ", ".join(str(int(i)) for i in unused[:10])
      raise ValueError(
        f"D={D}: Delaunay omitted {unused.size} input point(s); "
        f"first indices: {preview}"
      )

  simplices, determinants = orient_simplices_positive(
    X,
    simplices,
    determinant_tol,
  )
  face_to_elements, adjacency = build_face_incidence(D, simplices)
  validate_connected_graph(adjacency, set(range(simplices.shape[0])))

  return MeshData(
    D=D,
    X=np.ascontiguousarray(X, dtype=np.float64),
    simplices=simplices,
    face_to_elements=face_to_elements,
    adjacency=adjacency,
    simplex_determinants=determinants,
  )


def build_face_incidence(
  D: int,
  simplices: IntArray,
) -> tuple[dict[tuple[int, ...], tuple[int, ...]], Adjacency]:
  """Build canonical codimension-one face keys and the simplex dual graph."""
  owners: dict[tuple[int, ...], list[int]] = defaultdict(list)
  nelem = simplices.shape[0]
  adjacency: Adjacency = [set() for _ in range(nelem)]

  for elem, simplex in enumerate(simplices):
    if simplex.size != D + 1:
      raise ValueError(
        f"simplex {elem} has {simplex.size} vertices, expected {D + 1}"
      )
    if np.unique(simplex).size != D + 1:
      raise ValueError(f"simplex {elem} repeats a vertex")

    for opposite in range(D + 1):
      key = tuple(sorted(np.delete(simplex, opposite).tolist()))
      owners[key].append(elem)

  frozen_owners: dict[tuple[int, ...], tuple[int, ...]] = {}
  for key, face_owners in owners.items():
    if len(face_owners) > 2:
      raise ValueError(
        f"nonmanifold face {key} has {len(face_owners)} owning simplices"
      )
    frozen_owners[key] = tuple(face_owners)
    if len(face_owners) == 2:
      a, b = face_owners
      adjacency[a].add(b)
      adjacency[b].add(a)

  return frozen_owners, adjacency


def validate_connected_graph(adjacency: Adjacency, vertices: set[int]) -> None:
  if not vertices:
    raise ValueError("cannot validate an empty graph")

  start = min(vertices)
  stack = [start]
  visited = {start}

  while stack:
    vertex = stack.pop()
    for neighbor in adjacency[vertex]:
      if neighbor in vertices and neighbor not in visited:
        visited.add(neighbor)
        stack.append(neighbor)

  if visited != vertices:
    missing = sorted(vertices - visited)
    preview = ", ".join(str(i) for i in missing[:10])
    raise ValueError(
      f"induced dual graph is disconnected; {len(missing)} unreachable "
      f"element(s), first: {preview}"
    )


def has_cut_edge(left: set[int], right: set[int], adjacency: Adjacency) -> bool:
  if len(left) > len(right):
    left, right = right, left
  return any(neighbor in right for vertex in left for neighbor in adjacency[vertex])


def validate_split(
  parent: set[int],
  left: set[int],
  right: set[int],
  adjacency: Adjacency,
) -> bool:
  if not left or not right:
    return False
  if left & right:
    return False
  if left | right != parent:
    return False

  try:
    validate_connected_graph(adjacency, left)
    validate_connected_graph(adjacency, right)
  except ValueError:
    return False

  return has_cut_edge(left, right, adjacency)


def induced_local_adjacency(
  elements: Sequence[int],
  adjacency: Adjacency,
) -> list[list[int]]:
  local_index = {elem: i for i, elem in enumerate(elements)}
  return [
    sorted(local_index[nbr] for nbr in adjacency[elem] if nbr in local_index)
    for elem in elements
  ]


def _partition_from_labels(
  elements: set[int],
  ordered: Sequence[int],
  labels: Sequence[int],
  adjacency: Adjacency,
) -> tuple[set[int], set[int]] | None:
  """Convert a PyMetis label vector to a validated connected split."""
  if len(labels) != len(ordered):
    return None

  label_values = {int(label) for label in labels}
  if not label_values.issubset({0, 1}):
    return None

  left = {elem for elem, label in zip(ordered, labels) if int(label) == 0}
  right = elements - left
  if validate_split(elements, left, right, adjacency):
    return left, right
  return None


def pymetis_split(
  elements: set[int],
  adjacency: Adjacency,
  seed: int,
) -> tuple[set[int], set[int]]:
  """Return a validated connected PyMetis bisection.

  The outer Python routine already supplies the recursive tree construction.
  At each node we first ask METIS for its recursive two-way cut, which behaves
  well on tiny induced graphs.  If that cut is disconnected, we retry with
  the k-way algorithm and CONTIG/MINCONN enabled.  Only if both PyMetis cuts
  fail validation does the caller use the connected spanning-tree fallback.
  """
  try:
    import pymetis
  except ImportError as exc:
    raise RuntimeError(
      "PyMetis was requested, but importing pymetis failed under "
      f"interpreter {sys.executable!r}"
    ) from exc

  ordered = sorted(elements)
  local_adjacency = induced_local_adjacency(ordered, adjacency)
  module_path = getattr(pymetis, "__file__", "<unknown>")
  attempts: list[str] = []

  calls = [
    (
      "recursive",
      dict(
        recursive=True,
        options=pymetis.Options(seed=int(seed)),
      ),
    ),
    (
      "kway-contiguous",
      dict(
        recursive=False,
        options=pymetis.Options(
          seed=int(seed),
          contig=1,
          minconn=1,
        ),
      ),
    ),
  ]

  for mode, kwargs in calls:
    try:
      result = pymetis.part_graph(
        2,
        adjacency=local_adjacency,
        **kwargs,
      )
    except Exception as exc:
      attempts.append(f"{mode}: call failed: {exc!r}")
      continue

    labels = result.vertex_part if hasattr(result, "vertex_part") else result[1]
    split = _partition_from_labels(elements, ordered, labels, adjacency)
    if split is not None:
      return split

    attempts.append(
      f"{mode}: invalid labels={list(map(int, labels))}"
    )

  raise PyMetisPartitionError(
    "PyMetis did not produce a usable connected bisection for an induced "
    f"graph with {len(ordered)} vertices; module={module_path!r}; "
    + "; ".join(attempts)
  )


def spanning_tree_split(
  elements: set[int],
  adjacency: Adjacency,
) -> tuple[set[int], set[int]]:
  """Return a guaranteed connected bipartition using a DFS spanning tree cut."""
  if len(elements) < 2:
    raise ValueError("cannot split fewer than two elements")

  root = min(elements)
  parent: dict[int, int | None] = {root: None}
  children: dict[int, list[int]] = {vertex: [] for vertex in elements}
  order: list[int] = []
  stack: list[tuple[int, Iterable[int]]] = [
    (root, iter(sorted(adjacency[root] & elements)))
  ]
  order.append(root)

  while stack:
    vertex, neighbors = stack[-1]
    try:
      neighbor = next(neighbors)
    except StopIteration:
      stack.pop()
      continue

    if neighbor in parent:
      continue

    parent[neighbor] = vertex
    children[vertex].append(neighbor)
    order.append(neighbor)
    stack.append((neighbor, iter(sorted(adjacency[neighbor] & elements))))

  if len(parent) != len(elements):
    raise ValueError("fallback split received a disconnected induced graph")

  subtree_size = {vertex: 1 for vertex in elements}
  for vertex in reversed(order[1:]):
    parent_vertex = parent[vertex]
    assert parent_vertex is not None
    subtree_size[parent_vertex] += subtree_size[vertex]

  n = len(elements)
  split_root = min(
    (vertex for vertex in elements if vertex != root),
    key=lambda vertex: (
      abs(n - 2 * subtree_size[vertex]),
      -subtree_size[vertex],
      vertex,
    ),
  )

  left: set[int] = set()
  stack_vertices = [split_root]
  while stack_vertices:
    vertex = stack_vertices.pop()
    left.add(vertex)
    stack_vertices.extend(children[vertex])

  right = elements - left
  if not validate_split(elements, left, right, adjacency):
    raise RuntimeError("internal error: spanning-tree split is invalid")

  return left, right


def count_interface_faces(
  left: set[int],
  right: set[int],
  adjacency: Adjacency,
) -> int:
  if len(left) > len(right):
    left, right = right, left
  return sum(
    1
    for vertex in left
    for neighbor in adjacency[vertex]
    if neighbor in right
  )


def compute_tree_depths(
  nelem: int,
  merge_pairs: IntArray,
  root_id: int,
) -> dict[int, int]:
  children = {
    nelem + i: (int(pair[0]), int(pair[1]))
    for i, pair in enumerate(merge_pairs)
  }
  depths: dict[int, int] = {}
  stack = [(root_id, 0)]

  while stack:
    node, depth = stack.pop()
    depths[node] = depth
    if node >= nelem:
      left, right = children[node]
      stack.append((left, depth + 1))
      stack.append((right, depth + 1))

  return depths


def build_merge_tree(
  adjacency: Adjacency,
  partitioner: str,
  seed: int,
) -> TreeData:
  """Build bottom-up merge pairs with leaf IDs 0,...,nelem-1."""
  nelem = len(adjacency)
  if nelem < 1:
    raise ValueError("the mesh must contain at least one element")

  merge_pairs: list[tuple[int, int]] = []
  leaf_order: list[int] = []
  metis_splits = 0
  fallback_splits = 0
  child_imbalances: list[float] = []
  interface_sizes: list[int] = []

  def recurse(elements: set[int], recursion_seed: int) -> int:
    nonlocal metis_splits, fallback_splits

    if len(elements) == 1:
      leaf = next(iter(elements))
      leaf_order.append(leaf)
      return leaf

    if partitioner == "pymetis":
      try:
        split = pymetis_split(elements, adjacency, recursion_seed)
        metis_splits += 1
      except PyMetisPartitionError:
        split = spanning_tree_split(elements, adjacency)
        fallback_splits += 1
    elif partitioner == "fallback":
      split = spanning_tree_split(elements, adjacency)
      fallback_splits += 1
    else:
      raise ValueError(
        f"unknown partitioner {partitioner!r}; expected 'pymetis' or 'fallback'"
      )

    left, right = split
    interface_sizes.append(count_interface_faces(left, right, adjacency))
    child_imbalances.append(abs(len(left) - len(right)) / len(elements))

    left_root = recurse(left, recursion_seed + 1)
    right_root = recurse(right, recursion_seed + 104729)
    parent_id = nelem + len(merge_pairs)
    merge_pairs.append((left_root, right_root))
    return parent_id

  root_id = recurse(set(range(nelem)), seed)
  pairs = np.ascontiguousarray(merge_pairs, dtype=np.int32).reshape(-1, 2)
  validate_merge_tree(adjacency, pairs, root_id)

  depths = compute_tree_depths(nelem, pairs, root_id)
  leaf_depths = np.asarray([depths[i] for i in range(nelem)], dtype=np.float64)

  return TreeData(
    merge_pairs=pairs,
    root_id=root_id,
    leaf_order=np.asarray(leaf_order, dtype=np.int32),
    metis_splits=metis_splits,
    fallback_splits=fallback_splits,
    max_depth=int(np.max(leaf_depths)) if leaf_depths.size else 0,
    mean_leaf_depth=float(np.mean(leaf_depths)) if leaf_depths.size else 0.0,
    max_child_imbalance=max(child_imbalances, default=0.0),
    max_interface_faces=max(interface_sizes, default=0),
  )


def validate_merge_tree(
  adjacency: Adjacency,
  merge_pairs: IntArray,
  root_id: int,
) -> None:
  """Validate topology, disjointness, connectivity, and sibling interfaces."""
  nelem = len(adjacency)
  expected_merges = max(0, nelem - 1)
  if merge_pairs.shape != (expected_merges, 2):
    raise ValueError(
      f"merge_pairs has shape {merge_pairs.shape}, expected "
      f"({expected_merges}, 2)"
    )

  node_elements: dict[int, set[int]] = {
    leaf: {leaf} for leaf in range(nelem)
  }
  parent_count = {node: 0 for node in range(nelem + expected_merges)}

  for merge_index, pair in enumerate(merge_pairs):
    parent = nelem + merge_index
    left = int(pair[0])
    right = int(pair[1])

    if left == right:
      raise ValueError(f"merge {merge_index} repeats child node {left}")
    if left < 0 or right < 0 or left >= parent or right >= parent:
      raise ValueError(
        f"merge {merge_index} has non-topological children ({left}, {right}); "
        f"both must lie in [0, {parent})"
      )

    parent_count[left] += 1
    parent_count[right] += 1
    left_elements = node_elements[left]
    right_elements = node_elements[right]

    if left_elements & right_elements:
      overlap = sorted(left_elements & right_elements)
      raise ValueError(
        f"merge {merge_index} children overlap on elements {overlap[:10]}"
      )
    if not has_cut_edge(left_elements, right_elements, adjacency):
      raise ValueError(
        f"merge {merge_index} children do not share a codimension-one face"
      )

    combined = left_elements | right_elements
    validate_connected_graph(adjacency, combined)
    node_elements[parent] = combined

  expected_root = nelem + expected_merges - 1 if nelem > 1 else 0
  if root_id != expected_root:
    raise ValueError(f"root_id={root_id}, expected {expected_root}")
  if node_elements[root_id] != set(range(nelem)):
    raise ValueError("root node does not contain every leaf element exactly once")

  for node, count in parent_count.items():
    expected = 0 if node == root_id else 1
    if count != expected:
      raise ValueError(
        f"node {node} has parent count {count}, expected {expected}"
      )


def polynomial_dimensions(D: int, degree: int) -> tuple[int, int]:
  if degree < 0:
    raise ValueError("degree must be nonnegative")
  kf = math.comb(degree + D - 1, D - 1)
  vol_dim = math.comb(degree + D, D)
  return kf, vol_dim


class DummyHPSBackend:
  def __init__(self, library_path: Path, function_name: str) -> None:
    self.library_path = library_path
    self.function_name = function_name
    self.library = ctypes.CDLL(str(library_path))
    self.function = getattr(self.library, function_name)

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    self.function.argtypes = [
      ctypes.c_int,     # D
      ctypes.c_int,     # nverts
      c_double_p,       # X
      ctypes.c_int,     # nelem
      c_int_p,          # simplices
      ctypes.c_int,     # kf
      ctypes.c_int,     # vol_dim
      ctypes.c_int,     # nmerge
      c_int_p,          # merge_pairs
      ctypes.c_int,     # verbose
      c_double_p,       # root_robin_residual_inf
      c_double_p,       # interface_flux_residual_inf
      c_double_p,       # parent_consistency_residual_inf
      c_double_p,       # monolithic_trace_residual_inf
      c_double_p,       # leaf_volume_norm_inf
      c_int_p,          # root_nb
      c_int_p,          # nfaces
    ]
    self.function.restype = ctypes.c_int

  def run(
    self,
    mesh: MeshData,
    tree: TreeData,
    kf: int,
    vol_dim: int,
    verbose: int,
  ) -> BackendResult:
    X = np.ascontiguousarray(mesh.X, dtype=np.float64)
    simplices = np.ascontiguousarray(mesh.simplices, dtype=np.int32)
    merge_pairs = np.ascontiguousarray(tree.merge_pairs, dtype=np.int32)

    root_robin = ctypes.c_double()
    interface_flux = ctypes.c_double()
    parent_consistency = ctypes.c_double()
    monolithic_trace = ctypes.c_double()
    leaf_volume = ctypes.c_double()
    root_nb = ctypes.c_int()
    nfaces = ctypes.c_int()

    return_code = int(self.function(
      mesh.D,
      X.shape[0],
      X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
      simplices.shape[0],
      simplices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      int(kf),
      int(vol_dim),
      merge_pairs.shape[0],
      merge_pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      int(verbose),
      ctypes.byref(root_robin),
      ctypes.byref(interface_flux),
      ctypes.byref(parent_consistency),
      ctypes.byref(monolithic_trace),
      ctypes.byref(leaf_volume),
      ctypes.byref(root_nb),
      ctypes.byref(nfaces),
    ))

    result = BackendResult(
      return_code=return_code,
      root_robin_residual_inf=root_robin.value,
      interface_flux_residual_inf=interface_flux.value,
      parent_consistency_residual_inf=parent_consistency.value,
      monolithic_trace_residual_inf=monolithic_trace.value,
      leaf_volume_norm_inf=leaf_volume.value,
      root_nb=root_nb.value,
      nfaces=nfaces.value,
    )

    if return_code != 0:
      raise RuntimeError(
        f"{self.function_name} returned nonzero status {return_code} in D={mesh.D}"
      )

    return result


def export_case(
  output_dir: Path,
  mesh: MeshData,
  tree: TreeData,
) -> Path:
  output_path = output_dir / f"hps_mesh_tree_D{mesh.D}.npz"
  np.savez_compressed(
    output_path,
    D=np.asarray(mesh.D, dtype=np.int32),
    X=mesh.X,
    simplices=mesh.simplices,
    merge_pairs=tree.merge_pairs,
    root_id=np.asarray(tree.root_id, dtype=np.int32),
    leaf_order=tree.leaf_order,
    simplex_determinants=mesh.simplex_determinants,
  )
  return output_path


def unique_mesh_edges(simplices: IntArray) -> list[tuple[int, int]]:
  edges: set[tuple[int, int]] = set()
  for simplex in simplices:
    for a, b in combinations(simplex.tolist(), 2):
      edges.add((min(a, b), max(a, b)))
  return sorted(edges)


def element_rank(tree: TreeData, nelem: int) -> FloatArray:
  rank = np.empty(nelem, dtype=np.float64)
  rank[tree.leaf_order] = np.arange(nelem, dtype=np.float64)
  return rank


def visualize_case(
  output_dir: Path,
  mesh: MeshData,
  tree: TreeData,
  show: bool,
) -> Path | None:
  if mesh.D > 3:
    return None

  import matplotlib.pyplot as plt

  rank = element_rank(tree, mesh.simplices.shape[0])
  centroids = np.mean(mesh.X[mesh.simplices], axis=1)
  output_path = output_dir / f"hps_mesh_tree_D{mesh.D}.png"

  if mesh.D == 1:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    x = mesh.X[:, 0]
    for simplex in mesh.simplices:
      xa, xb = x[simplex]
      ax.plot((xa, xb), (0.0, 0.0), linewidth=2.0)
    ax.scatter(centroids[:, 0], np.zeros(centroids.shape[0]), c=rank, s=18)
    ax.scatter(x, np.zeros_like(x), marker="|", s=120)
    ax.set_ylim(-0.15, 0.15)
    ax.set_yticks([])
    ax.set_xlabel("x")

  elif mesh.D == 2:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.triplot(
      mesh.X[:, 0],
      mesh.X[:, 1],
      mesh.simplices,
      linewidth=0.6,
    )
    for a in range(len(mesh.adjacency)):
      for b in mesh.adjacency[a]:
        if a < b:
          ax.plot(
            centroids[[a, b], 0],
            centroids[[a, b], 1],
            linewidth=0.25,
            alpha=0.25,
          )
    ax.scatter(centroids[:, 0], centroids[:, 1], c=rank, s=9)
    ax.scatter(mesh.X[:, 0], mesh.X[:, 1], s=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

  else:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    mesh_segments = [
      mesh.X[[a, b], :]
      for a, b in unique_mesh_edges(mesh.simplices)
    ]
    dual_segments = [
      centroids[[a, b], :]
      for a in range(len(mesh.adjacency))
      for b in mesh.adjacency[a]
      if a < b
    ]
    ax.add_collection3d(
      Line3DCollection(mesh_segments, linewidths=0.35, alpha=0.22)
    )
    ax.add_collection3d(
      Line3DCollection(dual_segments, linewidths=0.25, alpha=0.18)
    )
    ax.scatter(
      centroids[:, 0],
      centroids[:, 1],
      centroids[:, 2],
      c=rank,
      s=8,
    )
    ax.scatter(mesh.X[:, 0], mesh.X[:, 1], mesh.X[:, 2], s=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

  partitioner_label = (
    f"PyMetis splits={tree.metis_splits}, fallback splits={tree.fallback_splits}"
  )
  ax.set_title(
    f"D={mesh.D}: {mesh.simplices.shape[0]} simplices, "
    f"tree depth={tree.max_depth}\n{partitioner_label}"
  )
  fig.tight_layout()
  fig.savefig(output_path, dpi=180, bbox_inches="tight")

  if show:
    plt.show()
  plt.close(fig)
  return output_path


def summarize_case(
  mesh: MeshData,
  tree: TreeData,
  mesh_file: Path,
  figure_file: Path | None,
  backend: BackendResult | None,
) -> DimensionSummary:
  owner_counts = np.asarray(
    [len(owners) for owners in mesh.face_to_elements.values()],
    dtype=np.int32,
  )
  ndual_edges = sum(len(neighbors) for neighbors in mesh.adjacency) // 2

  return DimensionSummary(
    D=mesh.D,
    nverts=mesh.X.shape[0],
    nelem=mesh.simplices.shape[0],
    nfaces=len(mesh.face_to_elements),
    nboundary_faces=int(np.count_nonzero(owner_counts == 1)),
    ninterior_faces=int(np.count_nonzero(owner_counts == 2)),
    ndual_edges=ndual_edges,
    min_abs_simplex_determinant=float(np.min(np.abs(mesh.simplex_determinants))),
    max_abs_simplex_determinant=float(np.max(np.abs(mesh.simplex_determinants))),
    root_id=tree.root_id,
    nmerge=tree.merge_pairs.shape[0],
    tree_max_depth=tree.max_depth,
    tree_mean_leaf_depth=tree.mean_leaf_depth,
    tree_max_child_imbalance=tree.max_child_imbalance,
    tree_max_interface_faces=tree.max_interface_faces,
    metis_splits=tree.metis_splits,
    fallback_splits=tree.fallback_splits,
    mesh_file=str(mesh_file),
    figure_file=str(figure_file) if figure_file is not None else None,
    backend=backend,
  )


def print_summary(summary: DimensionSummary) -> None:
  print(
    f"D={summary.D}: vertices={summary.nverts}, elements={summary.nelem}, "
    f"faces={summary.nfaces} "
    f"(boundary={summary.nboundary_faces}, interior={summary.ninterior_faces}), "
    f"dual_edges={summary.ndual_edges}"
  )
  print(
    f"  tree: merges={summary.nmerge}, root={summary.root_id}, "
    f"depth(max/mean)={summary.tree_max_depth}/"
    f"{summary.tree_mean_leaf_depth:.2f}, "
    f"max_imbalance={summary.tree_max_child_imbalance:.3f}, "
    f"max_interface_faces={summary.tree_max_interface_faces}, "
    f"splits(metis/fallback)={summary.metis_splits}/"
    f"{summary.fallback_splits}"
  )
  print(
    f"  simplex |det| range: "
    f"[{summary.min_abs_simplex_determinant:.3e}, "
    f"{summary.max_abs_simplex_determinant:.3e}]"
  )
  print(f"  data: {summary.mesh_file}")
  if summary.figure_file is not None:
    print(f"  figure: {summary.figure_file}")
  if summary.backend is not None:
    backend = summary.backend
    print(
      "  backend residuals: "
      f"root={backend.root_robin_residual_inf:.3e}, "
      f"interface={backend.interface_flux_residual_inf:.3e}, "
      f"parent={backend.parent_consistency_residual_inf:.3e}, "
      f"monolithic={backend.monolithic_trace_residual_inf:.3e}, "
      f"leaf_norm={backend.leaf_volume_norm_inf:.3e}, "
      f"root_nb={backend.root_nb}, nfaces={backend.nfaces}"
    )


def build_argument_parser() -> argparse.ArgumentParser:
  default_counts = ",".join(
    f"{D}:{count}" for D, count in DEFAULT_INTERIOR_COUNTS.items()
  )

  parser = argparse.ArgumentParser(
    description="Generate/validate D=1..5 simplex meshes and HPS merge trees."
  )
  parser.add_argument(
    "--dims",
    nargs="+",
    type=int,
    default=[1, 2, 3, 4, 5],
    help="dimensions to test (default: 1 2 3 4 5)",
  )
  parser.add_argument(
    "--interior-counts",
    type=parse_count_map,
    default=parse_count_map(default_counts),
    metavar="D:N,...",
    help=f"random interior point counts (default: {default_counts})",
  )
  parser.add_argument("--seed", type=int, default=20260715)
  parser.add_argument(
    "--partitioner",
    choices=("pymetis", "fallback"),
    default="pymetis",
    help=(
      "partition strategy (default: PyMetis preferred with validated "
      "connected fallback). Use fallback to skip PyMetis entirely."
    ),
  )
  parser.add_argument(
    "--qhull-options",
    default=DEFAULT_QHULL_OPTIONS,
    help=f"Qhull options for D>=2 (default: {DEFAULT_QHULL_OPTIONS!r})",
  )
  parser.add_argument(
    "--determinant-tol",
    type=float,
    default=1.0e-13,
    help="reject simplices with |affine determinant| at or below this value",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("hps_mesh_tree_output"),
  )
  parser.add_argument(
    "--no-viz",
    action="store_true",
    help="do not save D=1,2,3 mesh/dual-graph figures",
  )
  parser.add_argument(
    "--show",
    action="store_true",
    help="show D=1,2,3 figures interactively after saving",
  )
  parser.add_argument(
    "--lib",
    type=Path,
    help="optional shared library containing the dummy HPS entry point",
  )
  parser.add_argument(
    "--function",
    default="jhps_dummy_mesh_tree_test",
    help="C symbol to call when --lib is supplied",
  )
  parser.add_argument(
    "--degree",
    type=int,
    default=3,
    help="polynomial degree used to derive kf and vol_dim for the C call",
  )
  parser.add_argument(
    "--backend-verbose",
    type=int,
    default=1,
    help="verbose integer passed to the C backend",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_argument_parser()
  args = parser.parse_args(argv)

  dims = list(dict.fromkeys(args.dims))
  if any(D < 1 or D > 5 for D in dims):
    parser.error("this test driver supports dimensions 1 through 5")
  if args.determinant_tol < 0.0:
    parser.error("--determinant-tol must be nonnegative")
  if args.degree < 0:
    parser.error("--degree must be nonnegative")

  for D in dims:
    if D not in args.interior_counts:
      parser.error(
        f"--interior-counts does not specify D={D}; add an entry {D}:N"
      )

  args.output_dir.mkdir(parents=True, exist_ok=True)

  backend_driver: DummyHPSBackend | None = None
  if args.lib is not None:
    if not args.lib.exists():
      parser.error(f"shared library does not exist: {args.lib}")
    backend_driver = DummyHPSBackend(args.lib.resolve(), args.function)

  if args.partitioner == "pymetis":
    try:
      import pymetis
    except ImportError as exc:
      parser.error(
        "--partitioner pymetis requires PyMetis under "
        f"interpreter {sys.executable!r}: {exc}"
      )

    try:
      pymetis_version = importlib.metadata.version("pymetis")
    except importlib.metadata.PackageNotFoundError:
      pymetis_version = "unknown"
    pymetis_path = getattr(pymetis, "__file__", "<unknown>")
    print(
      "partitioner: pymetis "
      f"version={pymetis_version} module={pymetis_path}; "
      "validated fallback enabled"
    )
  else:
    print("partitioner: fallback; PyMetis disabled explicitly")

  summaries: list[DimensionSummary] = []
  seed_sequence = np.random.SeedSequence(args.seed)
  child_seeds = seed_sequence.spawn(len(dims))

  for D, child_seed in zip(dims, child_seeds):
    rng = np.random.default_rng(child_seed)
    mesh = generate_mesh(
      D=D,
      n_interior=args.interior_counts[D],
      rng=rng,
      qhull_options=args.qhull_options,
      determinant_tol=args.determinant_tol,
    )
    tree = build_merge_tree(
      adjacency=mesh.adjacency,
      partitioner=args.partitioner,
      seed=args.seed + 1009 * D,
    )

    mesh_file = export_case(args.output_dir, mesh, tree)
    figure_file = None
    if not args.no_viz:
      figure_file = visualize_case(args.output_dir, mesh, tree, args.show)

    backend_result = None
    if backend_driver is not None:
      kf, vol_dim = polynomial_dimensions(D, args.degree)
      backend_result = backend_driver.run(
        mesh=mesh,
        tree=tree,
        kf=kf,
        vol_dim=vol_dim,
        verbose=args.backend_verbose,
      )
      if backend_result.nfaces != len(mesh.face_to_elements):
        raise RuntimeError(
          f"D={D}: backend nfaces={backend_result.nfaces}, "
          f"Python nfaces={len(mesh.face_to_elements)}"
        )

    summary = summarize_case(
      mesh=mesh,
      tree=tree,
      mesh_file=mesh_file,
      figure_file=figure_file,
      backend=backend_result,
    )
    summaries.append(summary)
    print_summary(summary)

  summary_path = args.output_dir / "summary.json"
  with summary_path.open("w", encoding="utf-8") as stream:
    json.dump([asdict(summary) for summary in summaries], stream, indent=2)
    stream.write("\n")

  print(f"summary: {summary_path}")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except KeyboardInterrupt:
    print("interrupted", file=sys.stderr)
    raise SystemExit(130)
