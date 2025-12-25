import ctypes
import numpy as np
from libjpolyd_loader import libjpolyd

libjpolyd.jbasis_eval_all.argtypes = [
  ctypes.POINTER(ctypes.c_double),  # X
  ctypes.c_int,                     # ld_point
  ctypes.c_int,                     # ld_dim
  ctypes.c_int,                     # npts
  ctypes.POINTER(ctypes.c_double),  # kappa
  ctypes.c_int,                     # D
  ctypes.c_int,                     # n
  ctypes.POINTER(ctypes.c_int),     # alpha_table
  ctypes.POINTER(ctypes.c_int),     # tail_deg
  ctypes.POINTER(ctypes.c_double),  # inv_h
  ctypes.POINTER(ctypes.c_double),  # V
  ctypes.c_int,                     # ld_V
  ctypes.POINTER(ctypes.c_double),  # dV (may be NULL)
]
libjpolyd.jbasis_eval_all.restype = None

libjpolyd.jbasis_dim_Pi.argtypes = [ctypes.c_int, ctypes.c_int]
libjpolyd.jbasis_dim_Pi.restype = ctypes.c_int

libjpolyd.jbasis_build_structures.argtypes = [
  ctypes.POINTER(ctypes.c_double),  # kappa
  ctypes.c_int,                     # D
  ctypes.c_int,                     # n
  ctypes.POINTER(ctypes.c_int),     # alpha_table
  ctypes.POINTER(ctypes.c_int),     # tail_deg
  ctypes.POINTER(ctypes.c_double)   # inv_h
]
libjpolyd.jbasis_build_structures.restype = None

# ---- Python helpers ----

def jbasis_build_structures(D, n, kappa):
  """
  Build alpha_table, tail_deg, inv_h for Jacobi basis on D-simplex.
  
  Parameters
  ----------
  D : int
      Dimension of the simplex.
  n : int
      Maximum total degree.
  kappa : array_like, shape (D+1,)
      Jacobi parameters.
  
  Returns
  -------
  alpha_table : ndarray, shape (M, D), int32
      Multi-index table in graded lex order.
  tail_deg : ndarray, shape (M, D), int32
      Tail-degree table.
  inv_h : ndarray, shape (M,), float64
      1 / h_alpha for each basis function.
  """
  kappa = np.asarray(kappa, dtype=np.float64, order="C")
  if kappa.shape[0] != D + 1:
    raise ValueError("kappa must have length D+1")
  
  # Number of basis functions M = dim Pi_n^D
  M = libjpolyd.jbasis_dim_Pi(D, n)
  if M <= 0:
    raise ValueError("jbasis_dim_Pi returned non-positive M")
  
  alpha_table = np.zeros((M, D), dtype=np.int32, order="C")
  tail_deg    = np.zeros((M, D), dtype=np.int32, order="C")
  inv_h       = np.zeros((M,),   dtype=np.float64, order="C")
  
  kappa_ptr = kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  alpha_ptr = alpha_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  tail_ptr  = tail_deg.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  invh_ptr  = inv_h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  
  libjpolyd.jbasis_build_structures(
    kappa_ptr,
    D,
    n,
    alpha_ptr,
    tail_ptr,
    invh_ptr
  )
  
  return alpha_table, tail_deg, inv_h



def jbasis_eval_all(X, kappa, n, alpha_table, tail_deg, inv_h, D):
  """
  Python wrapper for jbasis_eval_all.
  
  X:          array of shape (npts, D)
  kappa:      array of shape (D+1,)
  n:          maximum total degree
  alpha_table: array of shape (M, D)
  tail_deg:    array of shape (M, D)
  inv_h:       array of shape (M,)
  D:          dimension (int)
  Returns:
    V: array of shape (npts, M), basis values.
  """
  X = np.asarray(X, dtype=np.float64, order="C")
  kappa = np.asarray(kappa, dtype=np.float64, order="C")
  alpha_table = np.asarray(alpha_table, dtype=np.int32, order="C")
  tail_deg = np.asarray(tail_deg, dtype=np.int32, order="C")
  inv_h = np.asarray(inv_h, dtype=np.float64, order="C")
  
  npts = X.shape[0]
  assert X.shape[1] == D
  M = libjpolyd.jbasis_dim_Pi(D, n)#alpha_table.shape[0]
  
  # Leading dimensions: X is AoS, so ld_point = D? no:
  # We used X[p*ld_point + j*ld_dim] in C.
  # For row-major (npts, D): ld_point = D, ld_dim = 1.
  ld_point = D
  ld_dim = 1
  
  # V is column-major in C: V[p + m*ld_V], ld_V = npts.
  V = np.zeros((npts, M), dtype=np.float64, order="F")
  ld_V = npts
  
  X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  kappa_ptr = kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  alpha_ptr = alpha_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  tail_ptr = tail_deg.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  inv_h_ptr = inv_h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  
  libjpolyd.jbasis_eval_all(
    X_ptr,
    ld_point,
    ld_dim,
    npts,
    kappa_ptr,
    D,
    n,
    alpha_ptr,
    tail_ptr,
    inv_h_ptr,
    V_ptr,
    ld_V,
    None
  )
  
  return V

def jbasis_eval_all_with_grad(X, kappa, n, alpha_table, tail_deg, inv_h, D):
  """
  Evaluate basis and gradients on the D-simplex.
  
  Parameters
  ----------
  X : array_like, shape (npts, D)
      Evaluation points.
  kappa : array_like, shape (D+1,)
      Jacobi parameters.
  n : int
      Maximum total degree.
  alpha_table : ndarray, shape (M, D)
  tail_deg : ndarray, shape (M, D)
  inv_h : ndarray, shape (M,)
  D : int
      Dimension.
  
  Returns
  -------
  V : ndarray, shape (npts, M)
      Basis values.
  dV : ndarray, shape (npts, M, D)
      Gradients, dV[p, m, ell] = d/dx_ell P_m(x_p).
  """
  X = np.asarray(X, dtype=np.float64, order="C")
  kappa = np.asarray(kappa, dtype=np.float64, order="C")
  alpha_table = np.asarray(alpha_table, dtype=np.int32, order="C")
  tail_deg = np.asarray(tail_deg, dtype=np.int32, order="C")
  inv_h = np.asarray(inv_h, dtype=np.float64, order="C")
  
  npts = X.shape[0]
  assert X.shape[1] == D
  M = libjpolyd.jbasis_dim_Pi(D, n)#alpha_table.shape[0]
  
  ld_point = D
  ld_dim = 1
  
  V = np.zeros((npts, M), dtype=np.float64, order="F")
  ld_V = npts
  
  # We store dV in shape (M, npts, D) to match the C layout:
  # index = ((m * ld_V) + p) * D + ell
  dV_raw = np.zeros((M, npts, D), dtype=np.float64, order="C")
  
  X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  kappa_ptr = kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  alpha_ptr = alpha_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  tail_ptr = tail_deg.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
  inv_h_ptr = inv_h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  dV_ptr = dV_raw.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  
  libjpolyd.jbasis_eval_all(
    X_ptr,
    ld_point,
    ld_dim,
    npts,
    kappa_ptr,
    D,
    n,
    alpha_ptr,
    tail_ptr,
    inv_h_ptr,
    V_ptr,
    ld_V,
    dV_ptr
  )
  
  # Reorder to (npts, M, D) for user convenience.
  dV = np.transpose(dV_raw, (1, 0, 2))
  
  return V, dV
