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
