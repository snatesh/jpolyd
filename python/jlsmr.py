import ctypes
import numpy as np

from libjpolyd_loader import libjpolyd


# int lsmr_dense_solve_colmajor(
#   int m, int n,
#   const double* A_colmajor,
#   const double* b,
#   double* x,
#   double damp,
#   double atol,
#   double btol,
#   double conlim,
#   int itnlim,
#   int nout,
#   int localsize,
#   int ctest,
#   int* istop_out,
#   int* itn_out,
#   int* stat_out,
#   double* normr_out,
#   double* normA_out,
#   double* condA_out,
#   double* normb_out,
#   double* normx_out,
#   double* normAr_out);
libjpolyd.lsmr_dense_solve_colmajor.argtypes = [
  ctypes.c_int,                             # m
  ctypes.c_int,                             # n
  ctypes.POINTER(ctypes.c_double),          # A_colmajor, shape (m,n), Fortran order
  ctypes.POINTER(ctypes.c_double),          # b, shape (m,)
  ctypes.POINTER(ctypes.c_double),          # x, shape (n,)
  ctypes.c_double,                          # damp
  ctypes.c_double,                          # atol
  ctypes.c_double,                          # btol
  ctypes.c_double,                          # conlim
  ctypes.c_int,                             # itnlim
  ctypes.c_int,                             # nout
  ctypes.c_int,                             # localsize
  ctypes.c_int,                             # ctest
  ctypes.POINTER(ctypes.c_int),             # istop_out
  ctypes.POINTER(ctypes.c_int),             # itn_out
  ctypes.POINTER(ctypes.c_int),             # stat_out
  ctypes.POINTER(ctypes.c_double),          # normr_out
  ctypes.POINTER(ctypes.c_double),          # normA_out
  ctypes.POINTER(ctypes.c_double),          # condA_out
  ctypes.POINTER(ctypes.c_double),          # normb_out
  ctypes.POINTER(ctypes.c_double),          # normx_out
  ctypes.POINTER(ctypes.c_double),          # normAr_out
]
libjpolyd.lsmr_dense_solve_colmajor.restype = ctypes.c_int


def lsmr_dense_solve(A, b, *, damp=0.0, atol=1.0e-12, btol=1.0e-12,
                     conlim=1.0e12, itnlim=500, nout=0,
                     localsize=0, ctest=3, return_info=True):
  """
  Solve min_x ||A x - b||_2 using the C/Fortran LSMR backend.

  Parameters
  ----------
  A : array_like, shape (m, n)
      Dense matrix. It is copied/coerced to float64 Fortran order.
  b : array_like, shape (m,)
      Right-hand side.
  damp : float
      LSMR damping parameter. Use 0.0 for ordinary least squares.
  atol, btol, conlim, itnlim, nout, localsize, ctest
      Passed through to the SOL LSMR routine.
      Set nout=6 for Fortran stdout diagnostics; nout=0 is quiet if supported.
  return_info : bool
      If True, return (x, info). If False, return x only.

  Returns
  -------
  x : ndarray, shape (n,)
  info : dict, optional
      Contains wrapper return code and LSMR diagnostics.
  """
  A = np.asarray(A, dtype=np.float64, order="F")
  if A.ndim != 2:
    raise ValueError("A must be 2D")
  if not A.flags.f_contiguous:
    A = np.asfortranarray(A, dtype=np.float64)

  b = np.asarray(b, dtype=np.float64, order="C")
  if b.ndim != 1:
    raise ValueError("b must be 1D")

  m, n = A.shape
  if b.size != m:
    raise ValueError(f"b must have length m={m}")

  x = np.zeros(n, dtype=np.float64)

  istop = ctypes.c_int(0)
  itn = ctypes.c_int(0)
  stat = ctypes.c_int(0)

  normr = ctypes.c_double(0.0)
  normA = ctypes.c_double(0.0)
  condA = ctypes.c_double(0.0)
  normb = ctypes.c_double(0.0)
  normx = ctypes.c_double(0.0)
  normAr = ctypes.c_double(0.0)

  ret = libjpolyd.lsmr_dense_solve_colmajor(
    ctypes.c_int(m),
    ctypes.c_int(n),
    A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_double(damp),
    ctypes.c_double(atol),
    ctypes.c_double(btol),
    ctypes.c_double(conlim),
    ctypes.c_int(itnlim),
    ctypes.c_int(nout),
    ctypes.c_int(localsize),
    ctypes.c_int(ctest),
    ctypes.byref(istop),
    ctypes.byref(itn),
    ctypes.byref(stat),
    ctypes.byref(normr),
    ctypes.byref(normA),
    ctypes.byref(condA),
    ctypes.byref(normb),
    ctypes.byref(normx),
    ctypes.byref(normAr),
  )

  info = {
    "ret": int(ret),
    "istop": int(istop.value),
    "itn": int(itn.value),
    "stat": int(stat.value),
    "normr": float(normr.value),
    "normA": float(normA.value),
    "condA": float(condA.value),
    "normb": float(normb.value),
    "normx": float(normx.value),
    "normAr": float(normAr.value),
  }

  if ret != 0:
    raise RuntimeError(f"lsmr_dense_solve_colmajor failed with wrapper code {ret}")
  if stat.value != 0:
    raise RuntimeError(f"LSMR reported allocation/status failure stat={stat.value}")

  if return_info:
    return x, info
  return x
