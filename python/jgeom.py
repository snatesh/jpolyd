import ctypes
import numpy as np

from libjpolyd_loader import libjpolyd


# -----------------------------
# C signatures
# -----------------------------

libjpolyd.jgeom_c_max_D.argtypes = []
libjpolyd.jgeom_c_max_D.restype = ctypes.c_int

libjpolyd.jgeom_invert_colmajor.argtypes = [
  ctypes.c_int,                              # D
  ctypes.POINTER(ctypes.c_double),           # A, length D*D, col-major
  ctypes.c_double,                           # rel_pivot_tol
  ctypes.POINTER(ctypes.c_double),           # Ainv, length D*D, col-major
  ctypes.POINTER(ctypes.c_double),           # det_out
]
libjpolyd.jgeom_invert_colmajor.restype = ctypes.c_int

libjpolyd.jgeom_affine_from_verts.argtypes = [
  ctypes.c_int,                              # D
  ctypes.POINTER(ctypes.c_double),           # V, shape D x (D+1), col-major
  ctypes.POINTER(ctypes.c_double),           # B_out, shape D x D, col-major
  ctypes.POINTER(ctypes.c_double),           # BinvT_out, shape D x D, col-major
  ctypes.POINTER(ctypes.c_double),           # detB_out
  ctypes.POINTER(ctypes.c_double),           # detBabs_out
]
libjpolyd.jgeom_affine_from_verts.restype = ctypes.c_int


def _check_ret(ret, name):
  if ret == 0:
    return
  if ret == 1:
    raise ValueError(f"{name}: null pointer passed to C")
  if ret == 2:
    raise ValueError(f"{name}: invalid or unsupported D")
  if ret == 3:
    raise np.linalg.LinAlgError(f"{name}: singular or numerically defective matrix/simplex")
  raise RuntimeError(f"{name}: failed with code {ret}")


def c_max_D():
  return int(libjpolyd.jgeom_c_max_D())


def invert_colmajor(A, rel_pivot_tol=0.0):
  """
  Invert a dense D x D matrix using the C++ column-major pivoted helper.

  Parameters
  ----------
  A : array_like, shape (D,D)
      Dense matrix. It is copied/coerced to Fortran order before crossing C.
  rel_pivot_tol : float, optional
      Relative pivot tolerance. If <= 0, the C++ default is used.

  Returns
  -------
  Ainv : ndarray, shape (D,D), Fortran-contiguous
  det : float
  """
  A = np.asarray(A, dtype=np.float64, order="F")
  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError("A must be a square 2D array")

  D = int(A.shape[0])
  Ainv = np.empty((D, D), dtype=np.float64, order="F")
  det = ctypes.c_double(0.0)

  ret = libjpolyd.jgeom_invert_colmajor(
    ctypes.c_int(D),
    A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_double(float(rel_pivot_tol)),
    Ainv.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.byref(det),
  )
  _check_ret(ret, "jgeom_invert_colmajor")
  return Ainv, float(det.value)


def affine_from_verts(V):
  """
  Build affine simplex geometry from a column-major vertex matrix.

  Parameters
  ----------
  V : array_like, shape (D,D+1)
      Vertex matrix with V[:, a] = vertex a. It is copied/coerced to
      Fortran order before crossing C.

  Returns
  -------
  info : dict
      Contains B, BinvT, detB, detBabs.
  """
  V = np.asarray(V, dtype=np.float64, order="F")
  if V.ndim != 2:
    raise ValueError("V must be a 2D array")

  D = int(V.shape[0])
  if V.shape[1] != D + 1:
    raise ValueError("V must have shape (D,D+1)")

  B = np.empty((D, D), dtype=np.float64, order="F")
  BinvT = np.empty((D, D), dtype=np.float64, order="F")
  detB = ctypes.c_double(0.0)
  detBabs = ctypes.c_double(0.0)

  ret = libjpolyd.jgeom_affine_from_verts(
    ctypes.c_int(D),
    V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    BinvT.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.byref(detB),
    ctypes.byref(detBabs),
  )
  _check_ret(ret, "jgeom_affine_from_verts")

  return {
    "B": B,
    "BinvT": BinvT,
    "detB": float(detB.value),
    "detBabs": float(detBabs.value),
  }
