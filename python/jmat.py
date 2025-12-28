import ctypes
import numpy as np
import scipy.sparse as sp
from libjpolyd_loader import libjpolyd

# --- C signatures ---

libjpolyd.jmat_dim_Pi.argtypes = [ctypes.c_int, ctypes.c_int]
libjpolyd.jmat_dim_Pi.restype  = ctypes.c_int

libjpolyd.jmat_build.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # kappa
    ctypes.c_int,                     # D
    ctypes.c_int,                     # n
    ctypes.POINTER(ctypes.c_double),  # J_all
]
libjpolyd.jmat_build.restype = ctypes.c_int

# int jmat_build_coord_pruned_csc(const double* kappa,
#                                 int D, int n, unsigned int nquad, int coord,
#                                 int** colptr_out, int** rowind_out, double** x_out,
#                                 int* N_out, int* nnz_out);
libjpolyd.jmat_build_coord_pruned_csc.argtypes = [
  ctypes.POINTER(ctypes.c_double),          # kappa
  ctypes.c_int,                             # D
  ctypes.c_int,                             # n
  ctypes.c_uint,                            # nquad
  ctypes.c_int,                             # coord
  ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),    # colptr_out
  ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),    # rowind_out
  ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), # x_out
  ctypes.POINTER(ctypes.c_int),             # N_out
  ctypes.POINTER(ctypes.c_int),             # nnz_out
]
libjpolyd.jmat_build_coord_pruned_csc.restype = ctypes.c_int

# void jmat_free(void* p);
libjpolyd.jmat_free.argtypes = [ctypes.c_void_p]
libjpolyd.jmat_free.restype  = None


def jmat_dim_Pi(D, n):
    """
    Return dim_Pi(n) for given dimension D.
    """
    N = libjpolyd.jmat_dim_Pi(int(D), int(n))
    if N <= 0:
        raise ValueError("jmat_dim_Pi failed for D={}, n={}".format(D, n))
    return N


def jmat_build(D, n, kappa):
    """
    Build Jacobi matrices J_i for i=1..D for Jacobi parameters kappa[0..D].

    Parameters
    ----------
    D : int
        Dimension of simplex.
    n : int
        Maximum total degree (J acts on Pi_n^D).
    kappa : array_like, shape (D+1,)
        Jacobi parameters (same convention as jbasis / jweight).

    Returns
    -------
    J_all : ndarray, shape (D, N, N)
        Jacobi matrices in row-major form, where N = dim_Pi(n).
        J_all[i, :, :] is the matrix for x_{i+1}.
    """
    kappa = np.asarray(kappa, dtype=np.float64, order="C")
    if kappa.shape[0] != D + 1:
        raise ValueError("kappa must have length D+1")

    N = jmat_dim_Pi(D, n)

    J_all = np.zeros((D, N, N), dtype=np.float64, order="C")

    kappa_ptr = kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    J_ptr     = J_all.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ret = libjpolyd.jmat_build(kappa_ptr, int(D), int(n), J_ptr)
    if ret != N:
        raise RuntimeError(
            "jmat_build failed: returned {}, expected N={}".format(ret, N)
        )

    return J_all


def _copy_and_free_csc(colptr_p, rowind_p, x_p, N, nnz):
  """
  Copy CSC arrays out of C-allocated memory into numpy arrays, then free C memory.
  """
  # Copy buffers into numpy
  colptr = np.ctypeslib.as_array(colptr_p, shape=(N + 1,)).copy()
  rowind = np.ctypeslib.as_array(rowind_p, shape=(nnz,)).copy()
  x      = np.ctypeslib.as_array(x_p,      shape=(nnz,)).copy()

  # Free C allocations
  libjpolyd.jmat_free(colptr_p)
  libjpolyd.jmat_free(rowind_p)
  libjpolyd.jmat_free(x_p)

  return colptr, rowind, x


def jmat_build_coord_csc(D, n, kappa, coord):
  """
  Build ONE coordinate Jacobi matrix J_coord in CSC format.

  Parameters
  ----------
  D : int
  n : int
  kappa : array_like, shape (D+1,)
  coord : int
      0..D-1

  Returns
  -------
  J : scipy.sparse.csc_matrix, shape (N,N)
  """
  D = int(D)
  n = int(n)
  coord = int(coord)

  if coord < 0 or coord >= D:
    raise ValueError("coord must be in 0..D-1")

  kappa = np.asarray(kappa, dtype=np.float64, order="C")
  if kappa.ndim != 1 or kappa.size != D + 1:
    raise ValueError("kappa must have length D+1")

  nquad = n + 1

  colptr_p = ctypes.POINTER(ctypes.c_int)()
  rowind_p = ctypes.POINTER(ctypes.c_int)()
  x_p      = ctypes.POINTER(ctypes.c_double)()

  N_out  = ctypes.c_int(0)
  nnz_out = ctypes.c_int(0)

  ret = libjpolyd.jmat_build_coord_pruned_csc(
    kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(D),
    ctypes.c_int(n),
    ctypes.c_uint(nquad),
    ctypes.c_int(coord),
    ctypes.byref(colptr_p),
    ctypes.byref(rowind_p),
    ctypes.byref(x_p),
    ctypes.byref(N_out),
    ctypes.byref(nnz_out),
  )
  if ret != 0:
    raise RuntimeError(f"jmat_build_coord_pruned_csc failed with code {ret}")

  N = int(N_out.value)
  nnz = int(nnz_out.value)
  if N <= 0:
    raise RuntimeError("jmat_build_coord_pruned_csc returned N<=0")
  if nnz < 0:
    raise RuntimeError("jmat_build_coord_pruned_csc returned nnz<0")

  colptr, rowind, x = _copy_and_free_csc(colptr_p, rowind_p, x_p, N, nnz)

  # Build CSC matrix; indices must be int32/64; SciPy accepts int32 fine.
  J = sp.csc_matrix((x, rowind, colptr), shape=(N, N))
  return J


def jmat_build_csc(D, n, kappa):
  """
  Build ALL coordinate Jacobi matrices as a list of CSC matrices.

  Returns
  -------
  J_list : list of scipy.sparse.csc_matrix length D
      J_list[i] corresponds to multiplication by x_i (i=0..D-1)
  """
  D = int(D)
  J_list = []
  for coord in range(D):
    J_list.append(jmat_build_coord_csc(D, n, kappa, coord))
  return J_list
