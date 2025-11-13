import numpy as np
import ctypes
from libjpolyd_loader import libjpolyd


libjpolyd.jquad_mapped_build.argtypes = [
    ctypes.c_int,                     # D
    ctypes.c_int,                     # n
    ctypes.POINTER(ctypes.c_double),  # X
    ctypes.POINTER(ctypes.c_double),  # W
]
libjpolyd.jquad_mapped_build.restype = ctypes.c_int

libjpolyd.jquad_mapped_npoints.argtypes = [ctypes.c_int, ctypes.c_uint]
libjpolyd.jquad_mapped_npoints.restype  = ctypes.c_int

libjpolyd.jquad_mapped_build_kappa.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # kappa
    ctypes.c_int,                     # D
    ctypes.c_uint,                    # n
    ctypes.POINTER(ctypes.c_double),  # points
    ctypes.POINTER(ctypes.c_double),  # weights
]
libjpolyd.jquad_mapped_build_kappa.restype = ctypes.c_int


def jquad_mapped_build(D, n):
    """
    Build mapped tensor-product quadrature for Lebesgue measure
    on the D-simplex.

    Parameters
    ----------
    D : int
        Dimension of the simplex (1 <= D <= compiled max).
    n : int
        Number of 1D nodes per axis. Total nodes npts = n**D.

    Returns
    -------
    X : np.ndarray, shape (npts, D)
        Quadrature nodes on the D-simplex.
    W : np.ndarray, shape (npts,)
        Quadrature weights. For f ≡ 1,
        sum(W) ~ vol(T^D) = 1 / D!.
    """
    if D <= 0:
        raise ValueError("D must be positive")
    if n <= 0:
        raise ValueError("n must be positive")

    npts = n ** D

    X = np.zeros((npts, D), dtype=np.float64)
    W = np.zeros(npts, dtype=np.float64)

    ret = libjpolyd.jquad_mapped_build(
        int(D),
        int(n),
        X.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    if ret != npts:
        raise RuntimeError(
            f"jquad_mapped_build failed, returned {ret}, expected {npts}"
        )

    return X, W


def jquad_mapped_build_kappa(D, n, kappa):
    kappa = np.asarray(kappa, dtype=np.float64)
    if kappa.shape[0] != D + 1:
        raise ValueError("kappa must have length D+1")

    N = libjpolyd.jquad_mapped_npoints(D, n)
    if N <= 0:
        raise ValueError("invalid (D, n) or overflow computing n^D")

    points = np.zeros((N, D), dtype=np.float64, order="C")
    weights = np.zeros((N,), dtype=np.float64, order="C")

    kappa_p  = kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    points_p = points.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    weights_p= weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ret = libjpolyd.jquad_mapped_build_kappa(kappa_p, D, n, points_p, weights_p)
    if ret != N:
        raise RuntimeError("jquad_mapped_build_kappa failed")

    return points, weights
