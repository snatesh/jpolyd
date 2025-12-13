import ctypes
import numpy as np

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

