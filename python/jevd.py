import ctypes
import numpy as np

from libjpolyd_loader import libjpolyd

# Signature:
# int jjevd_serial(double* J,
#                  int m, int n,
#                  double tol, int max_sweeps,
#                  int accumulate_V,
#                  double* V,
#                  int* sweeps_out,
#                  double* max_offdiag_out);

libjpolyd.jjevd_serial.argtypes = [
  ctypes.POINTER(ctypes.c_double),  # J
  ctypes.c_int,                     # m
  ctypes.c_int,                     # n
  ctypes.c_double,                  # tol
  ctypes.c_int,                     # max_sweeps
  ctypes.c_int,                     # accumulate_V
  ctypes.POINTER(ctypes.c_double),  # V (may be NULL)
  ctypes.POINTER(ctypes.c_int),     # sweeps_out
  ctypes.POINTER(ctypes.c_double),  # max_offdiag_out
]
libjpolyd.jjevd_serial.restype = ctypes.c_int


def jevd_serial(J, tol=1e-10, max_sweeps=50, accumulate_V=True):
  """
  Serial joint approximate EVD (Cardoso–Souloumiac) via jjevd_serial.

  Parameters
  ----------
  J : ndarray, shape (m, m, n) or (n, m, m)
      Stack of n real symmetric matrices.
      For best performance and exact layout match with the C code,
      use shape (m, m, n), Fortran order ('F'):
        J[i,j,nn] is entry (i,j) of matrix nn
        with column-major m x m per matrix.

  tol : float
      Threshold on |s| (rotation sine). Default 1e-10.

  max_sweeps : int
      Maximum sweeps over (p,q).

  accumulate_V : bool
      If True, also compute the joint diagonalizer V (m x m, column-major).

  Returns
  -------
  J_out : ndarray, shape (m, m, n), Fortran order
      The transformed matrices J' ≈ V^T J V, in-place result from C.
      Same layout (m, m, n, order='F').

  V : ndarray, shape (m, m), Fortran order
      Joint diagonalizer matrix. If accumulate_V=False, this is None.

  info : int
      0 if converged, 1 if not converged in max_sweeps, <0 for input error.

  sweeps : int
      Number of sweeps actually performed.

  max_offdiag : float
      Last max |s| (rotation sine) reported by the algorithm.
  """
  J = np.asarray(J, dtype=np.float64)

  if J.ndim != 3:
    raise ValueError("J must be a 3D array of shape (m,m,n) or (n,m,m)")

  # Accept either (m,m,n) or (n,m,m); normalize to (m,m,n)
  if J.shape[0] == J.shape[1]:
    # assume (m,m,n)
    m, m2, n = J.shape
    if m != m2:
      raise ValueError("J has shape (m,m,n) with non-square m x m")
    J_f = np.asfortranarray(J)  # ensure Fortran layout, shape (m,m,n)
  else:
    # assume (n,m,m) and transpose to (m,m,n)
    n, m, m2 = J.shape
    if m != m2:
      raise ValueError("J has shape (n,m,m) with non-square m x m")
    J_f = np.asfortranarray(J.transpose(1, 2, 0))  # (m,m,n), order='F'

  m, _, n = J_f.shape

  J_ptr = J_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

  if accumulate_V:
    V = np.eye(m, dtype=np.float64, order="F")
    V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  else:
    V = None
    V_ptr = ctypes.POINTER(ctypes.c_double)()  # NULL

  sweeps_c = ctypes.c_int(0)
  max_off_c = ctypes.c_double(0.0)

  info = libjpolyd.jjevd_serial(
    J_ptr,
    ctypes.c_int(m),
    ctypes.c_int(n),
    ctypes.c_double(tol),
    ctypes.c_int(max_sweeps),
    ctypes.c_int(1 if accumulate_V else 0),
    V_ptr,
    ctypes.byref(sweeps_c),
    ctypes.byref(max_off_c),
  )

  sweeps = sweeps_c.value
  max_offdiag = max_off_c.value

  # J_f is modified in-place; keep it as (m,m,n) Fortran.
  # You can return it directly, or transpose back if original was (n,m,m).
  # I'll always return (m,m,n) Fortran; the caller can reshape if desired.

  return J_f, V, info, sweeps, max_offdiag
