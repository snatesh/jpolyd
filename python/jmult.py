import ctypes
import numpy as np

# Reuse the same loader pattern as jmat_py.py
# (adjust import name to match your repo)
from libjpolyd_loader import libjpolyd


# -----------------------------
# C function signatures
# -----------------------------

libjpolyd.jmult_clenshaw_create.argtypes = [
  ctypes.POINTER(ctypes.c_double),  # kappa (len D+1)
  ctypes.c_int,                     # D
  ctypes.c_int,                     # p
  ctypes.c_int,                     # K
  ctypes.POINTER(ctypes.c_int),     # alpha_p (Mp*D row-major)
  ctypes.c_int,                     # Mp
  ctypes.c_int,                     # assume_symmetric (0/1)
  ctypes.POINTER(ctypes.c_void_p),  # handle_out
]
libjpolyd.jmult_clenshaw_create.restype = ctypes.c_int

libjpolyd.jmult_clenshaw_apply.argtypes = [
  ctypes.c_void_p,                  # handle
  ctypes.POINTER(ctypes.c_double),  # q (len Mp)
  ctypes.POINTER(ctypes.c_double),  # c (len MK)
  ctypes.POINTER(ctypes.c_double),  # y_out (len MK)
]
libjpolyd.jmult_clenshaw_apply.restype = ctypes.c_int

libjpolyd.jmult_clenshaw_destroy.argtypes = [ctypes.c_void_p]
libjpolyd.jmult_clenshaw_destroy.restype = None


# -----------------------------
# Python wrapper class
# -----------------------------

class JMultClenshaw:
  """
  Python wrapper for C-backed MultByQClenshaw (renamed jmult_* on the C side).

  Parameters
  ----------
  D : int
    Simplex dimension.
  p : int
    Degree of q expansion (Pi_p).
  K : int
    Degree of c expansion and output (Pi_K).
  kappa : array_like, shape (D+1,)
    Jacobi parameters.
  alpha_p : array_like, shape (Mp, D), dtype int32
    Alpha table for Pi_p basis ordering, row-major.
    Must be consistent with the C side's basis ordering.
  assume_symmetric : bool
    Match your python default (True) unless you know your Ji_p blocks are not symmetric.
  """

  def __init__(self, D, p, K, kappa, alpha_p, assume_symmetric=True):
    self.D = int(D)
    self.p = int(p)
    self.K = int(K)

    kappa = np.asarray(kappa, dtype=np.float64, order="C")
    if kappa.ndim != 1 or kappa.size != self.D + 1:
      raise ValueError(f"kappa must have shape ({self.D+1},)")

    alpha_p = np.asarray(alpha_p, dtype=np.int32, order="C")
    if alpha_p.ndim != 2 or alpha_p.shape[1] != self.D:
      raise ValueError(f"alpha_p must have shape (Mp, {self.D})")

    self.kappa = kappa
    self.alpha_p = alpha_p
    self.Mp = int(alpha_p.shape[0])

    handle = ctypes.c_void_p(None)
    ret = libjpolyd.jmult_clenshaw_create(
      kappa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
      ctypes.c_int(self.D),
      ctypes.c_int(self.p),
      ctypes.c_int(self.K),
      alpha_p.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
      ctypes.c_int(self.Mp),
      ctypes.c_int(1 if assume_symmetric else 0),
      ctypes.byref(handle),
    )
    if ret != 0 or not handle.value:
      raise RuntimeError(f"jmult_clenshaw_create failed with code {ret}")

    self._handle = handle

  def close(self):
    h = getattr(self, "_handle", None)
    if h is not None and h.value:
      libjpolyd.jmult_clenshaw_destroy(h)
      self._handle = ctypes.c_void_p(None)

  def __del__(self):
    # best-effort cleanup (avoid exceptions)
    try:
      self.close()
    except Exception:
      pass

  def apply(self, q, c, out=None):
    """
    Compute y = M_q c.

    q : (Mp,) float64
    c : (MK,) float64
    out : optional preallocated output array (MK,) float64

    Returns
    -------
    y : (MK,) float64
    """
    if not getattr(self, "_handle", None) or not self._handle.value:
      raise RuntimeError("Handle is closed")

    q = np.asarray(q, dtype=np.float64, order="C")
    if q.ndim != 1 or q.size != self.Mp:
      raise ValueError(f"q must have shape ({self.Mp},)")

    c = np.asarray(c, dtype=np.float64, order="C")
    if c.ndim != 1:
      raise ValueError("c must be 1D")

    if out is None:
      y = np.empty_like(c)
    else:
      y = np.asarray(out, dtype=np.float64, order="C")
      if y.shape != c.shape:
        raise ValueError("out must have same shape as c")

    ret = libjpolyd.jmult_clenshaw_apply(
      self._handle,
      q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
      c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
      y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if ret != 0:
      raise RuntimeError(f"jmult_clenshaw_apply failed with code {ret}")

    return y

