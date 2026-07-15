import ctypes
import numpy as np
import scipy.sparse as sp

from libjpolyd_loader import libjpolyd


# int js_kmat_build_tprod(int D, int n, unsigned int q,
#                         const double* kappa_src,
#                         const double* kappa_tgt,
#                         double* K_out);
libjpolyd.js_kmat_build_tprod.argtypes = [
  ctypes.c_int,                      # D
  ctypes.c_int,                      # n
  ctypes.c_uint,                     # q
  ctypes.POINTER(ctypes.c_double),   # kappa_src
  ctypes.POINTER(ctypes.c_double),   # kappa_tgt
  ctypes.POINTER(ctypes.c_double),   # K_out
]
libjpolyd.js_kmat_build_tprod.restype = ctypes.c_int


import math
def _dim_Pi(D, n):
  D = int(D)
  n = int(n)
  if D < 1 or n < 0:
    raise ValueError("invalid D or n")
  return math.comb(n + D, D)



def kmat_build_tprod(D, n, q, kappa_src, kappa_tgt):
  """
  Build dense promotion matrix K using internal κ-aware mapped tensor-product quadrature.

  K maps coeffs in basis(kappa_src) -> coeffs in basis(kappa_tgt), using L2(w_kappa_tgt).

  Parameters
  ----------
  D : int
    Simplex dimension (2 or 3 for your current use).
  n : int
    Max total degree of basis.
  q : int
    1D Gauss-Jacobi points per axis in mapped quadrature (rule exactness ~ 2q-1).
    A good default is q = n+1.
  kappa_src : array_like, shape (D+1,)
  kappa_tgt : array_like, shape (D+1,)

  Returns
  -------
  K : ndarray, shape (M, M), row-major
  """
  D = int(D)
  n = int(n)
  q = int(q)
  if n < 0:
    raise ValueError("n must be >= 0")
  if q <= 0:
    raise ValueError("q must be >= 1")

  kappa_src = np.asarray(kappa_src, dtype=np.float64, order="C")
  kappa_tgt = np.asarray(kappa_tgt, dtype=np.float64, order="C")
  if kappa_src.ndim != 1 or kappa_src.size != D + 1:
    raise ValueError("kappa_src must have shape (D+1,)")
  if kappa_tgt.ndim != 1 or kappa_tgt.size != D + 1:
    raise ValueError("kappa_tgt must have shape (D+1,)")

  M = _dim_Pi(D, n)
  K = np.zeros((M, M), dtype=np.float64, order="C")

  ret = libjpolyd.js_kmat_build_tprod(
    ctypes.c_int(D),
    ctypes.c_int(n),
    ctypes.c_uint(q),
    kappa_src.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    kappa_tgt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    K.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
  )
  if ret != 0:
    raise RuntimeError(f"js_kmat_build_tprod failed with code {ret}")

  return K

# int js_kmat_build_tprod_pruned_csc(int D, int n, unsigned int q,
#                                   const double* kappa_src,
#                                   const double* kappa_tgt,
#                                   int* nrow_out, int* ncol_out, int* nnz_out,
#                                   int** colptr_out, int** rowind_out, double** x_out);
libjpolyd.js_kmat_build_tprod_pruned_csc.argtypes = [
  ctypes.c_int,                                         # D
  ctypes.c_int,                                         # n
  ctypes.c_uint,                                        # q
  ctypes.POINTER(ctypes.c_double),                      # kappa_src
  ctypes.POINTER(ctypes.c_double),                      # kappa_tgt
  ctypes.POINTER(ctypes.c_int),                         # nrow_out
  ctypes.POINTER(ctypes.c_int),                         # ncol_out
  ctypes.POINTER(ctypes.c_int),                         # nnz_out
  ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),         # colptr_out
  ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),         # rowind_out
  ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),      # x_out
]
libjpolyd.js_kmat_build_tprod_pruned_csc.restype = ctypes.c_int

# void js_kmat_csc_free(int* colptr, int* rowind, double* x);
libjpolyd.js_kmat_csc_free.argtypes = [
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_int),
  ctypes.POINTER(ctypes.c_double),
]
libjpolyd.js_kmat_csc_free.restype = None

def kmat_build_tprod_pruned_csc(D, n, q, kappa_src, kappa_tgt, *, dtype=np.float64):
  """
  Build sparse promotion/connection matrix K in CSC form.

  Returns
  -------
  K : scipy.sparse.csc_matrix, shape (M, M), M=dim_Pi(D,n)

  Notes
  -----
  The C library allocates (colptr,rowind,x) with malloc; we copy into NumPy-owned
  arrays and free the C buffers via js_kmat_csc_free.
  """
  D = int(D)
  n = int(n)
  q = int(q)

  if n < 0:
    raise ValueError("n must be >= 0")
  if q <= 0:
    raise ValueError("q must be >= 1")

  kappa_src = np.asarray(kappa_src, dtype=np.float64, order="C")
  kappa_tgt = np.asarray(kappa_tgt, dtype=np.float64, order="C")
  if kappa_src.ndim != 1 or kappa_src.size != D + 1:
    raise ValueError("kappa_src must have shape (D+1,)")
  if kappa_tgt.ndim != 1 or kappa_tgt.size != D + 1:
    raise ValueError("kappa_tgt must have shape (D+1,)")

  nrow_c = ctypes.c_int(0)
  ncol_c = ctypes.c_int(0)
  nnz_c  = ctypes.c_int(0)

  colptr_p = ctypes.POINTER(ctypes.c_int)()
  rowind_p = ctypes.POINTER(ctypes.c_int)()
  x_p      = ctypes.POINTER(ctypes.c_double)()

  ret = libjpolyd.js_kmat_build_tprod_pruned_csc(
    ctypes.c_int(D),
    ctypes.c_int(n),
    ctypes.c_uint(q),
    kappa_src.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    kappa_tgt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.byref(nrow_c),
    ctypes.byref(ncol_c),
    ctypes.byref(nnz_c),
    ctypes.byref(colptr_p),
    ctypes.byref(rowind_p),
    ctypes.byref(x_p),
  )
  if ret != 0:
    raise RuntimeError(f"js_kmat_build_tprod_pruned_csc failed with code {ret}")

  nrow = int(nrow_c.value)
  ncol = int(ncol_c.value)
  nnz  = int(nnz_c.value)

  try:
    indptr = np.ctypeslib.as_array(colptr_p, shape=(ncol + 1,)).astype(np.int32, copy=True)

    if nnz > 0:
      indices = np.ctypeslib.as_array(rowind_p, shape=(nnz,)).astype(np.int32, copy=True)
      data = np.ctypeslib.as_array(x_p, shape=(nnz,)).astype(np.float64, copy=True)
    else:
      indices = np.empty((0,), dtype=np.int32)
      data = np.empty((0,), dtype=np.float64)

  finally:
    libjpolyd.js_kmat_csc_free(colptr_p, rowind_p, x_p)

  if dtype is not None and dtype != np.float64:
    data = data.astype(dtype, copy=False)

  return sp.csc_matrix((data, indices, indptr), shape=(nrow, ncol))

