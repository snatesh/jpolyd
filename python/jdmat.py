import ctypes
import numpy as np

try:
  import scipy.sparse as sp
except ImportError as e:
  raise ImportError("jdmat.py CSC bindings require scipy (scipy.sparse)") from e

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

# int js_dmat_build_tprod_natural_pruned_csc(int D, int n, unsigned int q,
#                                           const double* kappa_src,
#                                           int axis,
#                                           int* nrow_out, int* ncol_out, int* nnz_out,
#                                           int** colptr_out, int** rowind_out, double** x_out);
libjpolyd.js_dmat_build_tprod_natural_pruned_csc.argtypes = [
  ctypes.c_int,                        # D
  ctypes.c_int,                        # n
  ctypes.c_uint,                       # q
  ctypes.POINTER(ctypes.c_double),     # kappa_src
  ctypes.c_int,                        # axis
  ctypes.POINTER(ctypes.c_int),        # nrow_out
  ctypes.POINTER(ctypes.c_int),        # ncol_out
  ctypes.POINTER(ctypes.c_int),        # nnz_out
  ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),     # colptr_out
  ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),     # rowind_out
  ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # x_out
]
libjpolyd.js_dmat_build_tprod_natural_pruned_csc.restype = ctypes.c_int

# void js_dmat_csc_free(int* colptr, int* rowind, double* x);
libjpolyd.js_dmat_csc_free.argtypes = [
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_double),
]
libjpolyd.js_dmat_csc_free.restype = None


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

def dmat_build_tprod_natural_pruned_csc(D, n, q, kappa_src, axis, *, dtype=np.float64):
  """
  Build the "natural" differentiation operator in CSC format.

  Returns:
    scipy.sparse.csc_matrix with shape (dim_Pi(D,n-1), dim_Pi(D,n)) for n>=1.
    For n==0: shape (0, dim_Pi(D,0)) == (0,1), nnz=0.

  Memory ownership:
    C allocates colptr,rowind,x; we copy into NumPy arrays and then free C buffers.
  """
  D = int(D)
  n = int(n)
  q = int(q)
  axis = int(axis)

  if n < 0:
    raise ValueError("n must be >= 0")
  if q <= 0:
    raise ValueError("q must be >= 1")

  kappa_src = np.asarray(kappa_src, dtype=np.float64, order="C")
  if kappa_src.ndim != 1 or kappa_src.size != D + 1:
    raise ValueError("kappa_src must have shape (D+1,)")

  nrow_c = ctypes.c_int(0)
  ncol_c = ctypes.c_int(0)
  nnz_c = ctypes.c_int(0)

  colptr_p = ctypes.POINTER(ctypes.c_int)()
  rowind_p = ctypes.POINTER(ctypes.c_int)()
  x_p = ctypes.POINTER(ctypes.c_double)()

  ret = libjpolyd.js_dmat_build_tprod_natural_pruned_csc(
    ctypes.c_int(D),
    ctypes.c_int(n),
    ctypes.c_uint(q),
    kappa_src.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(axis),
    ctypes.byref(nrow_c),
    ctypes.byref(ncol_c),
    ctypes.byref(nnz_c),
    ctypes.byref(colptr_p),
    ctypes.byref(rowind_p),
    ctypes.byref(x_p),
  )
  if ret != 0:
    raise RuntimeError(f"js_dmat_build_tprod_natural_pruned_csc failed with code {ret}")

  nrow = int(nrow_c.value)
  ncol = int(ncol_c.value)
  nnz = int(nnz_c.value)

  try:
    # Wrap C buffers as NumPy views (no copy yet)
    # colptr length is ncol+1 even when nnz==0
    colptr_view = np.ctypeslib.as_array(colptr_p, shape=(ncol + 1,)).astype(np.int32, copy=True)

    if nnz > 0:
      rowind_view = np.ctypeslib.as_array(rowind_p, shape=(nnz,)).astype(np.int32, copy=True)
      x_view = np.ctypeslib.as_array(x_p, shape=(nnz,)).astype(np.float64, copy=True)
    else:
      rowind_view = np.empty((0,), dtype=np.int32)
      x_view = np.empty((0,), dtype=np.float64)

  finally:
    # Always free C memory, even if copy construction fails
    libjpolyd.js_dmat_csc_free(colptr_p, rowind_p, x_p)

  # Build scipy CSC
  # SciPy expects indptr (colptr) dtype int32/int64; indices dtype int32/int64; data float
  if dtype is not None and dtype != np.float64:
    x_view = x_view.astype(dtype, copy=False)

  A = sp.csc_matrix((x_view, rowind_view, colptr_view), shape=(nrow, ncol))
  return A

