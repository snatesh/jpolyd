import ctypes
import numpy as np
from libjpolyd_loader import libjpolyd

# double jweight_w_kappa(const double* kappa, int D);
libjpolyd.jweight_w_kappa.argtypes = [ctypes.POINTER(ctypes.c_double),\
                                       ctypes.c_int]
libjpolyd.jweight_w_kappa.restype = ctypes.c_double

# void jweight_eval(const double* X, int ld_point, int ld_dim, int npts,
#                   const double* kappa, double* out, int D);
libjpolyd.jweight_eval.argtypes = [ctypes.POINTER(ctypes.c_double),  # X
                                    ctypes.c_int,                     # ld_point
                                    ctypes.c_int,                     # ld_dim
                                    ctypes.c_int,                     # npts
                                    ctypes.POINTER(ctypes.c_double),  # kappa
                                    ctypes.POINTER(ctypes.c_double),  # out
                                    ctypes.c_int]                     # D
libjpolyd.jweight_eval.restype = None


def jweight_w_kappa(kappa: np.ndarray, D: int) -> float:
  kappa = np.ascontiguousarray(kappa, dtype=np.float64)
  kappa_ptr = kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  return float(libjpolyd.jweight_w_kappa(kappa_ptr, int(D)))


def jweight_eval(X: np.ndarray, ld_point: int, ld_dim: int, npts: int,
                 kappa: np.ndarray, D: int, out: np.ndarray | None = None) -> np.ndarray:
  # Ensure correct dtype and contiguity for all arrays
  X     = np.ascontiguousarray(X,     dtype=np.float64)
  kappa = np.ascontiguousarray(kappa, dtype=np.float64)

  if out is None:
    out = np.zeros(int(npts), dtype=np.float64)
  else:
    # Validate provided output buffer
    assert out.dtype == np.float64 and out.size >= int(npts)
    if not out.flags['C_CONTIGUOUS']:
      out = np.ascontiguousarray(out)

  # Marshalled pointers
  X_ptr     = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  kappa_ptr = kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  out_ptr   = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

  # Call C API (AoS version)
  libjpolyd.jweight_eval(X_ptr, int(ld_point), int(ld_dim), int(npts),
                          kappa_ptr, out_ptr, int(D))
  return out
