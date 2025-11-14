import ctypes
import numpy as np

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
