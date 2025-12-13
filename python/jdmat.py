import ctypes
import numpy as np

from libjpolyd_loader import libjpolyd


# int js_dmat_build_tprod(int D, int n, unsigned int q,
#                         const double* kappa_src,
#                         const double* kappa_rng,
#                         int axis,
#                         double* D_out);
libjpolyd.js_dmat_build_tprod.argtypes = [
  ctypes.c_int,                      # D
  ctypes.c_int,                      # n
  ctypes.c_uint,                     # q
  ctypes.POINTER(ctypes.c_double),   # kappa_src
  ctypes.POINTER(ctypes.c_double),   # kappa_rng
  ctypes.c_int,                      # axis
  ctypes.POINTER(ctypes.c_double),   # D_out
]
libjpolyd.js_dmat_build_tprod.restype = ctypes.c_int

# int js_dmat_build_tprod_natural_pruned(int D, int n, unsigned int q,
#                                        const double* kappa_src,
#                                        int axis,
#                                        double* D_out);
libjpolyd.js_dmat_build_tprod_natural_pruned.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_uint,
  ctypes.POINTER(ctypes.c_double),
  ctypes.c_int,
  ctypes.POINTER(ctypes.c_double),
]
libjpolyd.js_dmat_build_tprod_natural_pruned.restype = ctypes.c_int

def _dim_Pi(D, n):
  D = int(D)
  n = int(n)
  if n < 0:
    raise ValueError("n must be >= 0")
  if D == 1:
    return n + 1
  if D == 2:
    return (n + 1) * (n + 2) // 2
  if D == 3:
    return (n + 1) * (n + 2) * (n + 3) // 6
  if D == 4:
    return (n + 1) * (n + 2) * (n + 3) * (n + 4) // 24
  raise ValueError("Unsupported D (compiled library may support fewer)")


def dmat_build_tprod(D, n, q, kappa_src, kappa_rng, axis):
  """
  Build dense derivative projection matrix using internal mapped κ-quadrature.

  Output:
    Dmat (M,M) row-major, M=dim_Pi(D,n)
  """
  D = int(D)
  n = int(n)
  q = int(q)
  axis = int(axis)

  if n < 1:
    raise ValueError("n must be >= 1")
  if q <= 0:
    raise ValueError("q must be >= 1")

  kappa_src = np.asarray(kappa_src, dtype=np.float64, order="C")
  kappa_rng = np.asarray(kappa_rng, dtype=np.float64, order="C")

  if kappa_src.ndim != 1 or kappa_src.size != D + 1:
    raise ValueError("kappa_src must have shape (D+1,)")
  if kappa_rng.ndim != 1 or kappa_rng.size != D + 1:
    raise ValueError("kappa_rng must have shape (D+1,)")

  M = _dim_Pi(D, n)
  Dmat = np.zeros((M, M), dtype=np.float64, order="C")

  ret = libjpolyd.js_dmat_build_tprod(
    ctypes.c_int(D),
    ctypes.c_int(n),
    ctypes.c_uint(q),
    kappa_src.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    kappa_rng.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(axis),
    Dmat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  if ret != 0:
    raise RuntimeError(f"js_dmat_build_tprod failed with code {ret}")

  return Dmat


def dmat_build_tprod_natural_pruned(D, n, q, kappa_src, axis):
  D = int(D)
  n = int(n)
  q = int(q)
  axis = int(axis)

  kappa_src = np.asarray(kappa_src, dtype=np.float64, order="C")
  if kappa_src.ndim != 1 or kappa_src.size != D + 1:
    raise ValueError("kappa_src must have shape (D+1,)")

  M = _dim_Pi(D, n)
  Dmat = np.zeros((M, M), dtype=np.float64, order="C")

  ret = libjpolyd.js_dmat_build_tprod_natural_pruned(
    ctypes.c_int(D),
    ctypes.c_int(n),
    ctypes.c_uint(q),
    kappa_src.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(axis),
    Dmat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  if ret != 0:
    raise RuntimeError(f"js_dmat_build_tprod_natural_pruned failed with code {ret}")

  return Dmat

