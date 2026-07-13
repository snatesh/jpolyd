import ctypes
import numpy as np

from libjpolyd_loader import libjpolyd


# -----------------------------
# C signatures
# -----------------------------

libjpolyd.jperms_c_max_D.argtypes = []
libjpolyd.jperms_c_max_D.restype = ctypes.c_int

libjpolyd.jperms_face_vertices.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_int),
]
libjpolyd.jperms_face_vertices.restype = ctypes.c_int

libjpolyd.jperms_face_sigma_array.argtypes = [
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_int),
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_int),
]
libjpolyd.jperms_face_sigma_array.restype = ctypes.c_int

libjpolyd.jperms_face_sigma_index.argtypes = [
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_int),
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_int),
]
libjpolyd.jperms_face_sigma_index.restype = ctypes.c_int

libjpolyd.jperms_perm_to_lehmer_index.argtypes = [
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_int),
]
libjpolyd.jperms_perm_to_lehmer_index.restype = ctypes.c_int

libjpolyd.jperms_face_values_local_to_canonical_double.argtypes = [
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_double),
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_double),
]
libjpolyd.jperms_face_values_local_to_canonical_double.restype = ctypes.c_int

libjpolyd.jperms_kappa_storage_to_vertex.argtypes = [
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_double),
  ctypes.POINTER(ctypes.c_double),
]
libjpolyd.jperms_kappa_storage_to_vertex.restype = ctypes.c_int

libjpolyd.jperms_kappa_vertex_to_storage.argtypes = [
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_double),
  ctypes.POINTER(ctypes.c_double),
]
libjpolyd.jperms_kappa_vertex_to_storage.restype = ctypes.c_int

libjpolyd.jperms_common_face_kappa.argtypes = [
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_double),
  ctypes.POINTER(ctypes.c_double),
]
libjpolyd.jperms_common_face_kappa.restype = ctypes.c_int

libjpolyd.jperms_dropped_face_kappa_canonical.argtypes = [
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_double),
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_double),
]
libjpolyd.jperms_dropped_face_kappa_canonical.restype = ctypes.c_int


def _check_ret(ret, name):
  if ret == 0:
    return
  if ret == 1:
    raise ValueError(f"{name}: null pointer passed to C")
  if ret == 2:
    raise ValueError(f"{name}: invalid or unsupported D")
  if ret == 4:
    raise ValueError(f"{name}: invalid face_id")
  if ret == 5:
    raise ValueError(f"{name}: invalid sigma permutation")
  raise RuntimeError(f"{name}: failed with code {ret}")


def _as_i32_1d(a, size=None, name="array"):
  a = np.asarray(a, dtype=np.int32, order="C")
  if a.ndim != 1:
    raise ValueError(f"{name} must be 1D")
  if size is not None and a.size != size:
    raise ValueError(f"{name} must have length {size}")
  return a


def _as_f64_1d(a, size=None, name="array"):
  a = np.asarray(a, dtype=np.float64, order="C")
  if a.ndim != 1:
    raise ValueError(f"{name} must be 1D")
  if size is not None and a.size != size:
    raise ValueError(f"{name} must have length {size}")
  return a


def c_max_D():
  return int(libjpolyd.jperms_c_max_D())


def face_vertices(D, face_id):
  D = int(D)
  out = np.empty(D, dtype=np.int32)
  ret = libjpolyd.jperms_face_vertices(
    ctypes.c_int(D),
    ctypes.c_int(int(face_id)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
  )
  _check_ret(ret, "jperms_face_vertices")
  return out


def face_sigma_array(global_vids, face_id):
  global_vids = _as_i32_1d(global_vids, name="global_vids")
  D = int(global_vids.size - 1)
  sigma = np.empty(D, dtype=np.int32)
  ret = libjpolyd.jperms_face_sigma_array(
    ctypes.c_int(D),
    global_vids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.c_int(int(face_id)),
    sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
  )
  _check_ret(ret, "jperms_face_sigma_array")
  return sigma


def face_sigma_index(global_vids, face_id):
  global_vids = _as_i32_1d(global_vids, name="global_vids")
  D = int(global_vids.size - 1)
  out = ctypes.c_int(0)
  ret = libjpolyd.jperms_face_sigma_index(
    ctypes.c_int(D),
    global_vids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.c_int(int(face_id)),
    ctypes.byref(out),
  )
  _check_ret(ret, "jperms_face_sigma_index")
  return int(out.value)


def perm_to_lehmer_index(sigma):
  sigma = _as_i32_1d(sigma, name="sigma")
  D = int(sigma.size)
  out = ctypes.c_int(0)
  ret = libjpolyd.jperms_perm_to_lehmer_index(
    ctypes.c_int(D),
    sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.byref(out),
  )
  _check_ret(ret, "jperms_perm_to_lehmer_index")
  return int(out.value)


def face_values_local_to_canonical(values_local, sigma):
  values_local = _as_f64_1d(values_local, name="values_local")
  sigma = _as_i32_1d(sigma, size=values_local.size, name="sigma")
  D = int(values_local.size)
  values_canonical = np.empty(D, dtype=np.float64)
  ret = libjpolyd.jperms_face_values_local_to_canonical_double(
    ctypes.c_int(D),
    values_local.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    values_canonical.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  _check_ret(ret, "jperms_face_values_local_to_canonical_double")
  return values_canonical


def kappa_storage_to_vertex(kappa_storage):
  kappa_storage = _as_f64_1d(kappa_storage, name="kappa_storage")
  D = int(kappa_storage.size - 1)
  out = np.empty(D + 1, dtype=np.float64)
  ret = libjpolyd.jperms_kappa_storage_to_vertex(
    ctypes.c_int(D),
    kappa_storage.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  _check_ret(ret, "jperms_kappa_storage_to_vertex")
  return out


def kappa_vertex_to_storage(kappa_vertex):
  kappa_vertex = _as_f64_1d(kappa_vertex, name="kappa_vertex")
  D = int(kappa_vertex.size - 1)
  out = np.empty(D + 1, dtype=np.float64)
  ret = libjpolyd.jperms_kappa_vertex_to_storage(
    ctypes.c_int(D),
    kappa_vertex.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  _check_ret(ret, "jperms_kappa_vertex_to_storage")
  return out


def common_face_kappa(kappa_volume):
  kappa_volume = _as_f64_1d(kappa_volume, name="kappa_volume")
  D = int(kappa_volume.size - 1)
  out = np.empty(D, dtype=np.float64)
  ret = libjpolyd.jperms_common_face_kappa(
    ctypes.c_int(D),
    kappa_volume.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  _check_ret(ret, "jperms_common_face_kappa")
  return out


def dropped_face_kappa_canonical(kappa_volume, face_id, sigma_local_to_canonical):
  kappa_volume = _as_f64_1d(kappa_volume, name="kappa_volume")
  D = int(kappa_volume.size - 1)
  sigma_local_to_canonical = _as_i32_1d(
    sigma_local_to_canonical,
    size=D,
    name="sigma_local_to_canonical",
  )
  out = np.empty(D, dtype=np.float64)
  ret = libjpolyd.jperms_dropped_face_kappa_canonical(
    ctypes.c_int(D),
    kappa_volume.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(int(face_id)),
    sigma_local_to_canonical.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  _check_ret(ret, "jperms_dropped_face_kappa_canonical")
  return out
