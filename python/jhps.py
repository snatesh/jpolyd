from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass, field

import numpy as np

from libjpolyd_loader import libjpolyd


_Int32C = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_Float64C = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


@dataclass
class HpsDummyResult:
  D: int
  root_robin_residual_inf: float
  interface_flux_residual_inf: float
  parent_consistency_residual_inf: float
  monolithic_trace_residual_inf: float
  leaf_volume_norm_inf: float
  root_nb: int
  interface_nb: int


@dataclass
class HpsPoissonResult:
  D: int
  n: int
  nelem: int
  M: int
  m_int: int
  kf: int
  root_nb: int
  interface_nb: int
  root_robin_residual_inf: float
  interface_flux_residual_inf: float
  parent_consistency_residual_inf: float
  leaf_coeffs: np.ndarray = field(repr=False)


def _set_common_signature(fn) -> None:
  fn.argtypes = [
    ct.c_int,       # D
    ct.c_int,       # nverts
    _Int32C,        # vertex_ids
    _Float64C,      # coords_rowmajor
    ct.c_int,       # nelem
    _Int32C,        # simplices_rowmajor
    ct.c_int,       # kf
    ct.c_int,       # vol_dim
    ct.c_uint,      # seed
    ct.c_double,    # alpha
    ct.c_double,    # beta
    ct.c_int,       # verbose
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_int),
    ct.POINTER(ct.c_int),
  ]
  fn.restype = ct.c_int


def _set_mesh_tree_signature(fn) -> None:
  fn.argtypes = [
    ct.c_int,       # D
    ct.c_int,       # nverts
    _Int32C,        # vertex_ids
    _Float64C,      # coords_rowmajor
    ct.c_int,       # nelem
    _Int32C,        # simplices_rowmajor
    ct.c_int,       # nmerge
    _Int32C,        # merge_pairs_rowmajor
    ct.c_int,       # kf
    ct.c_int,       # vol_dim
    ct.c_uint,      # seed
    ct.c_double,    # alpha
    ct.c_double,    # beta
    ct.c_int,       # verbose
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_int),
    ct.POINTER(ct.c_int),
  ]
  fn.restype = ct.c_int


def _set_poisson_mesh_tree_signature(fn) -> None:
  fn.argtypes = [
    ct.c_int,       # D
    ct.c_int,       # n
    ct.c_int,       # q_pad
    ct.c_int,       # q_vol
    ct.c_int,       # q_face
    _Float64C,      # kappa, D+1
    ct.c_int,       # nverts
    _Int32C,        # vertex_ids
    _Float64C,      # coords_rowmajor
    ct.c_int,       # nelem
    _Int32C,        # simplices_rowmajor
    ct.c_int,       # nmerge
    _Int32C,        # merge_pairs_rowmajor
    _Float64C,      # f_int_elementmajor
    ct.c_int,       # nboundary_faces
    _Int32C,        # boundary_face_keys_rowmajor
    _Float64C,      # boundary_g_rowmajor
    ct.c_double,    # tau_C
    ct.c_double,    # alpha
    ct.c_double,    # beta
    ct.c_int,       # verbose
    _Float64C,      # leaf_coeffs_elementmajor
    ct.POINTER(ct.c_double),  # root residual
    ct.POINTER(ct.c_double),  # interface residual
    ct.POINTER(ct.c_double),  # parent residual
    ct.POINTER(ct.c_int),     # M_out
    ct.POINTER(ct.c_int),     # m_int_out
    ct.POINTER(ct.c_int),     # kf_out
    ct.POINTER(ct.c_int),     # root_nb_out
    ct.POINTER(ct.c_int),     # interface_nb_out
  ]
  fn.restype = ct.c_int


for _name in (
  "jhps_dummy_two_leaf_test",
  "jhps_dummy_three_leaf_chain_test",
  "jhps_dummy_four_leaf_balanced_test",
):
  _set_common_signature(getattr(libjpolyd, _name))

_set_mesh_tree_signature(libjpolyd.jhps_dummy_mesh_tree_test)
_set_poisson_mesh_tree_signature(libjpolyd.jhps_poisson_mesh_tree_solve)


def load_library() -> ct.CDLL:
  """Compatibility shim: the project library is loaded by libjpolyd_loader."""
  return libjpolyd


def _validate_mesh_tree_arrays(
  D: int,
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplices: np.ndarray,
  merge_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  vertex_ids = np.ascontiguousarray(vertex_ids, dtype=np.int32)
  coords = np.ascontiguousarray(coords, dtype=np.float64)
  simplices = np.ascontiguousarray(simplices, dtype=np.int32)
  merge_pairs = np.ascontiguousarray(merge_pairs, dtype=np.int32)

  if vertex_ids.ndim != 1:
    raise ValueError("vertex_ids must have shape (nverts,)")
  if vertex_ids.size < D + 1:
    raise ValueError("not enough vertices for one D-simplex")
  if np.unique(vertex_ids).size != vertex_ids.size:
    raise ValueError("vertex_ids must be unique")
  if coords.shape != (vertex_ids.size, D):
    raise ValueError(
      f"coords must have shape ({vertex_ids.size}, {D}), got {coords.shape}"
    )
  if simplices.ndim != 2 or simplices.shape[1] != D + 1:
    raise ValueError(
      f"simplices must have shape (nelem, {D + 1}), got {simplices.shape}"
    )

  nelem = int(simplices.shape[0])
  if nelem < 1:
    raise ValueError("simplices must contain at least one element")
  if not np.all(np.isin(simplices, vertex_ids)):
    raise ValueError("simplices contains a vertex ID not present in vertex_ids")

  expected_nmerge = nelem - 1
  if merge_pairs.size == 0:
    merge_pairs = np.empty((0, 2), dtype=np.int32)
  if merge_pairs.shape != (expected_nmerge, 2):
    raise ValueError(
      f"merge_pairs must have shape ({expected_nmerge}, 2), "
      f"got {merge_pairs.shape}"
    )

  for m, pair in enumerate(merge_pairs):
    parent_id = nelem + m
    a = int(pair[0])
    b = int(pair[1])
    if a == b:
      raise ValueError(f"merge {m} repeats child {a}")
    if a < 0 or b < 0 or a >= parent_id or b >= parent_id:
      raise ValueError(
        f"merge {m} children {(a, b)} must lie in [0, {parent_id})"
      )

  return vertex_ids, coords, simplices, merge_pairs


def run_poisson_mesh_tree_solve(
  pc,
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplices: np.ndarray,
  merge_pairs: np.ndarray,
  f_int_elementmajor: np.ndarray,
  boundary_face_keys: np.ndarray,
  boundary_g: np.ndarray,
  *,
  tau_C: float = 10.0,
  alpha: float = 1.0,
  beta: float = 0.0,
  verbose: bool = False,
) -> HpsPoissonResult:
  """Solve Poisson on an external simplex mesh and external HPS merge tree.

  ``pc`` is a Python ``RefSimplexPrecomp`` object.  It is used only for its
  dimensions and constructor parameters; the C++ solver constructs the same
  reference precompute internally.

  Array layouts
  -------------
  f_int_elementmajor : (nelem, pc.m_int), float64 C-order
  boundary_face_keys : (nboundary_faces, D), int32 C-order
  boundary_g         : (nboundary_faces, pc.kf), float64 C-order

  ``boundary_g`` contains canonical common-face projections of
  ``alpha*u + beta*du_dn``.  The C wrapper reorders face blocks by the supplied
  sorted global face keys.
  """
  D = int(pc.D)
  n = int(pc.n)
  if D < 1 or D > 5:
    raise ValueError("D must be in 1..5")
  if n < 2:
    raise ValueError("Poisson HPS requires n >= 2")
  if float(alpha) == 0.0:
    raise ValueError(
      "the current C wrapper requires alpha != 0; pure Neumann is not implemented"
    )
  if not np.isfinite(tau_C) or float(tau_C) <= 0.0:
    raise ValueError("tau_C must be finite and positive")

  vertex_ids, coords, simplices, merge_pairs = _validate_mesh_tree_arrays(
    D, vertex_ids, coords, simplices, merge_pairs
  )
  nelem = int(simplices.shape[0])
  nmerge = int(merge_pairs.shape[0])

  kappa = np.ascontiguousarray(pc.kappa, dtype=np.float64)
  if kappa.shape != (D + 1,):
    raise ValueError(f"pc.kappa must have shape ({D + 1},)")

  f_int = np.ascontiguousarray(f_int_elementmajor, dtype=np.float64)
  if f_int.shape != (nelem, int(pc.m_int)):
    raise ValueError(
      f"f_int_elementmajor must have shape ({nelem}, {pc.m_int}), "
      f"got {f_int.shape}"
    )

  boundary_face_keys = np.ascontiguousarray(boundary_face_keys, dtype=np.int32)
  boundary_g = np.ascontiguousarray(boundary_g, dtype=np.float64)
  if boundary_face_keys.ndim != 2 or boundary_face_keys.shape[1] != D:
    raise ValueError(
      f"boundary_face_keys must have shape (nboundary_faces, {D}), "
      f"got {boundary_face_keys.shape}"
    )
  nboundary_faces = int(boundary_face_keys.shape[0])
  if nboundary_faces < 1:
    raise ValueError("boundary_face_keys must contain at least one boundary face")
  if boundary_g.shape != (nboundary_faces, int(pc.kf)):
    raise ValueError(
      f"boundary_g must have shape ({nboundary_faces}, {pc.kf}), "
      f"got {boundary_g.shape}"
    )

  canonical_keys = np.sort(boundary_face_keys, axis=1)
  if np.unique(canonical_keys, axis=0).shape[0] != nboundary_faces:
    raise ValueError("boundary_face_keys contains duplicate canonical face keys")
  boundary_face_keys = np.ascontiguousarray(canonical_keys, dtype=np.int32)

  leaf_coeffs = np.empty((nelem, int(pc.M)), dtype=np.float64, order="C")
  root_res = ct.c_double()
  iface_res = ct.c_double()
  parent_res = ct.c_double()
  M_out = ct.c_int()
  m_int_out = ct.c_int()
  kf_out = ct.c_int()
  root_nb = ct.c_int()
  interface_nb = ct.c_int()

  rc = libjpolyd.jhps_poisson_mesh_tree_solve(
    ct.c_int(D),
    ct.c_int(n),
    ct.c_int(int(pc.q_pad)),
    ct.c_int(int(pc.q_vol)),
    ct.c_int(int(pc.q_face)),
    kappa,
    ct.c_int(vertex_ids.size),
    vertex_ids,
    coords,
    ct.c_int(nelem),
    simplices,
    ct.c_int(nmerge),
    merge_pairs,
    f_int,
    ct.c_int(nboundary_faces),
    boundary_face_keys,
    boundary_g,
    ct.c_double(float(tau_C)),
    ct.c_double(float(alpha)),
    ct.c_double(float(beta)),
    ct.c_int(1 if verbose else 0),
    leaf_coeffs,
    ct.byref(root_res),
    ct.byref(iface_res),
    ct.byref(parent_res),
    ct.byref(M_out),
    ct.byref(m_int_out),
    ct.byref(kf_out),
    ct.byref(root_nb),
    ct.byref(interface_nb),
  )
  if rc != 0:
    raise RuntimeError(f"jhps_poisson_mesh_tree_solve failed with rc={rc}")

  returned_dims = (M_out.value, m_int_out.value, kf_out.value)
  expected_dims = (int(pc.M), int(pc.m_int), int(pc.kf))
  if returned_dims != expected_dims:
    raise RuntimeError(
      "Python/C++ RefSimplexPrecomp dimension mismatch: "
      f"C++ returned {returned_dims}, Python has {expected_dims}"
    )

  return HpsPoissonResult(
    D=D,
    n=n,
    nelem=nelem,
    M=M_out.value,
    m_int=m_int_out.value,
    kf=kf_out.value,
    root_nb=root_nb.value,
    interface_nb=interface_nb.value,
    root_robin_residual_inf=root_res.value,
    interface_flux_residual_inf=iface_res.value,
    parent_consistency_residual_inf=parent_res.value,
    leaf_coeffs=leaf_coeffs,
  )


def chain_simplex_mesh(D: int, nelem: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  D = int(D)
  nelem = int(nelem)
  if D < 1 or nelem < 1:
    raise ValueError("D and nelem must be positive")

  nverts = D + nelem
  vertex_ids = np.arange(nverts, dtype=np.int32)
  coords = np.empty((nverts, D), dtype=np.float64, order="C")
  for v in range(nverts):
    t = float(v + 1)
    p = t
    for r in range(D):
      coords[v, r] = p
      p *= t

  simplices = np.empty((nelem, D + 1), dtype=np.int32, order="C")
  for e in range(nelem):
    simplices[e, :] = np.arange(e, e + D + 1, dtype=np.int32)
  return vertex_ids, coords, simplices


def _run_fixed_dummy_test(
  fn_name: str,
  D: int,
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplices: np.ndarray,
  *,
  kf: int,
  vol_dim: int,
  seed: int,
  alpha: float,
  beta: float,
  verbose: bool,
) -> HpsDummyResult:
  vertex_ids = np.ascontiguousarray(vertex_ids, dtype=np.int32)
  coords = np.ascontiguousarray(coords, dtype=np.float64)
  simplices = np.ascontiguousarray(simplices, dtype=np.int32)

  root_res = ct.c_double()
  iface_res = ct.c_double()
  parent_res = ct.c_double()
  mono_res = ct.c_double()
  vol_norm = ct.c_double()
  root_nb = ct.c_int()
  interface_nb = ct.c_int()

  rc = getattr(libjpolyd, fn_name)(
    ct.c_int(D), ct.c_int(vertex_ids.size), vertex_ids, coords,
    ct.c_int(simplices.shape[0]), simplices,
    ct.c_int(kf), ct.c_int(vol_dim), ct.c_uint(seed),
    ct.c_double(alpha), ct.c_double(beta), ct.c_int(1 if verbose else 0),
    ct.byref(root_res), ct.byref(iface_res), ct.byref(parent_res),
    ct.byref(mono_res), ct.byref(vol_norm), ct.byref(root_nb),
    ct.byref(interface_nb),
  )
  if rc != 0:
    raise RuntimeError(f"{fn_name} failed with rc={rc}")

  return HpsDummyResult(
    D=D,
    root_robin_residual_inf=root_res.value,
    interface_flux_residual_inf=iface_res.value,
    parent_consistency_residual_inf=parent_res.value,
    monolithic_trace_residual_inf=mono_res.value,
    leaf_volume_norm_inf=vol_norm.value,
    root_nb=root_nb.value,
    interface_nb=interface_nb.value,
  )


def run_mesh_tree_test(
  D: int,
  vertex_ids: np.ndarray,
  coords: np.ndarray,
  simplices: np.ndarray,
  merge_pairs: np.ndarray,
  *,
  kf: int = 2,
  vol_dim: int | None = None,
  seed: int = 123,
  alpha: float = 1.0,
  beta: float = 1.0,
  verbose: bool = False,
  lib: ct.CDLL | None = None,
) -> HpsDummyResult:
  del lib
  if vol_dim is None:
    vol_dim = max(int(D) + 2, 2 * int(kf))
  vertex_ids, coords, simplices, merge_pairs = _validate_mesh_tree_arrays(
    int(D), vertex_ids, coords, simplices, merge_pairs
  )

  root_res = ct.c_double()
  iface_res = ct.c_double()
  parent_res = ct.c_double()
  mono_res = ct.c_double()
  vol_norm = ct.c_double()
  root_nb = ct.c_int()
  interface_nb = ct.c_int()

  rc = libjpolyd.jhps_dummy_mesh_tree_test(
    ct.c_int(int(D)), ct.c_int(vertex_ids.size), vertex_ids, coords,
    ct.c_int(simplices.shape[0]), simplices,
    ct.c_int(merge_pairs.shape[0]), merge_pairs,
    ct.c_int(int(kf)), ct.c_int(int(vol_dim)), ct.c_uint(int(seed)),
    ct.c_double(float(alpha)), ct.c_double(float(beta)),
    ct.c_int(1 if verbose else 0),
    ct.byref(root_res), ct.byref(iface_res), ct.byref(parent_res),
    ct.byref(mono_res), ct.byref(vol_norm), ct.byref(root_nb),
    ct.byref(interface_nb),
  )
  if rc != 0:
    raise RuntimeError(f"jhps_dummy_mesh_tree_test failed with rc={rc}")

  return HpsDummyResult(
    D=int(D),
    root_robin_residual_inf=root_res.value,
    interface_flux_residual_inf=iface_res.value,
    parent_consistency_residual_inf=parent_res.value,
    monolithic_trace_residual_inf=mono_res.value,
    leaf_volume_norm_inf=vol_norm.value,
    root_nb=root_nb.value,
    interface_nb=interface_nb.value,
  )


def run_two_leaf_test(D: int, *, kf: int = 2, vol_dim: int | None = None,
                      seed: int = 123, alpha: float = 1.0, beta: float = 1.0,
                      verbose: bool = False, lib=None) -> HpsDummyResult:
  del lib
  if vol_dim is None:
    vol_dim = max(D + 2, 2 * kf)
  vids, X, E = chain_simplex_mesh(D, 2)
  return _run_fixed_dummy_test(
    "jhps_dummy_two_leaf_test", D, vids, X, E, kf=kf, vol_dim=vol_dim,
    seed=seed, alpha=alpha, beta=beta, verbose=verbose)


def run_three_leaf_chain_test(D: int, *, kf: int = 2, vol_dim: int | None = None,
                              seed: int = 123, alpha: float = 1.0,
                              beta: float = 1.0, verbose: bool = False,
                              lib=None) -> HpsDummyResult:
  del lib
  if vol_dim is None:
    vol_dim = max(D + 2, 2 * kf)
  vids, X, E = chain_simplex_mesh(D, 3)
  return _run_fixed_dummy_test(
    "jhps_dummy_three_leaf_chain_test", D, vids, X, E, kf=kf,
    vol_dim=vol_dim, seed=seed, alpha=alpha, beta=beta, verbose=verbose)


def run_four_leaf_balanced_test(D: int, *, kf: int = 2,
                                vol_dim: int | None = None, seed: int = 123,
                                alpha: float = 1.0, beta: float = 1.0,
                                verbose: bool = False,
                                lib=None) -> HpsDummyResult:
  del lib
  if vol_dim is None:
    vol_dim = max(D + 2, 2 * kf)
  vids, X, E = chain_simplex_mesh(D, 4)
  return _run_fixed_dummy_test(
    "jhps_dummy_four_leaf_balanced_test", D, vids, X, E, kf=kf,
    vol_dim=vol_dim, seed=seed, alpha=alpha, beta=beta, verbose=verbose)
