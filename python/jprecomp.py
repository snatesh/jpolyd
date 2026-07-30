import ctypes
import math
import numpy as np

from libjpolyd_loader import libjpolyd

_double_p = ctypes.POINTER(ctypes.c_double)
_int_p = ctypes.POINTER(ctypes.c_int)
_void_pp = ctypes.POINTER(ctypes.c_void_p)

libjpolyd.jprecomp_create.argtypes = [
  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
  _double_p, _void_pp,
]
libjpolyd.jprecomp_create.restype = ctypes.c_int

libjpolyd.jprecomp_destroy.argtypes = [ctypes.c_void_p]
libjpolyd.jprecomp_destroy.restype = None

libjpolyd.jprecomp_dims.argtypes = [
  ctypes.c_void_p,
  _int_p, _int_p, _int_p, _int_p, _int_p, _int_p,
  _int_p, _int_p, _int_p, _int_p, _int_p,
]
libjpolyd.jprecomp_dims.restype = ctypes.c_int

libjpolyd.jprecomp_get_face_ref_scale.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_face_ref_scale.restype = ctypes.c_int

libjpolyd.jprecomp_get_kappa_res.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_kappa_res.restype = ctypes.c_int

libjpolyd.jprecomp_get_Lij_ref.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_Lij_ref.restype = ctypes.c_int

libjpolyd.jprecomp_get_Li_ref.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_Li_ref.restype = ctypes.c_int

libjpolyd.jprecomp_get_L0_ref.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_L0_ref.restype = ctypes.c_int

libjpolyd.jprecomp_get_T_ref.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_T_ref.restype = ctypes.c_int

libjpolyd.jprecomp_get_Fgrad_ref.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_Fgrad_ref.restype = ctypes.c_int

libjpolyd.jprecomp_get_Mface_ref.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_Mface_ref.restype = ctypes.c_int

libjpolyd.jprecomp_get_volume_quad.argtypes = [ctypes.c_void_p, _double_p, _double_p]
libjpolyd.jprecomp_get_volume_quad.restype = ctypes.c_int

libjpolyd.jprecomp_get_volume_basis.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_volume_basis.restype = ctypes.c_int

libjpolyd.jprecomp_get_residual_quad.argtypes = [ctypes.c_void_p, _double_p, _double_p]
libjpolyd.jprecomp_get_residual_quad.restype = ctypes.c_int

libjpolyd.jprecomp_get_residual_basis.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_residual_basis.restype = ctypes.c_int

libjpolyd.jprecomp_get_face_quad.argtypes = [ctypes.c_void_p, _double_p, _double_p]
libjpolyd.jprecomp_get_face_quad.restype = ctypes.c_int

libjpolyd.jprecomp_get_face_basis.argtypes = [ctypes.c_void_p, _double_p]
libjpolyd.jprecomp_get_face_basis.restype = ctypes.c_int

def _ptr_f64(a):
  return a.ctypes.data_as(_double_p)


def _check(ret, name):
  if ret == 0:
    return
  if ret == 1:
    raise ValueError(f"{name}: null pointer")
  if ret == 2:
    raise ValueError(f"{name}: invalid dimensions or arguments")
  if ret == 3:
    raise ValueError(f"{name}: unsupported D")
  if ret == 4:
    raise RuntimeError(f"{name}: C++ exception during precompute")
  raise RuntimeError(f"{name}: unknown return code {ret}")


def dimPi(D, n):
  if n < 0:
    return 0
  return math.comb(n + D, D)


def num_face_perms(D):
  return math.factorial(int(D))


class RefSimplexPrecomp:
  """ctypes wrapper for C++ RefSimplexPrecomp<D,double>."""

  def __init__(self, D, n, kappa, q_pad=2, q_vol=None, q_face=None):
    self.D = int(D)
    self.n = int(n)
    self.q_pad = int(q_pad)
    if q_vol is None:
      q_vol = 0
    if q_face is None:
      q_face = 0
    self.q_vol_requested = int(q_vol)
    self.q_face_requested = int(q_face)

    kappa = np.asarray(kappa, dtype=np.float64, order="C")
    if kappa.shape != (self.D + 1,):
      raise ValueError(f"kappa must have shape ({self.D + 1},)")
    self.kappa = kappa

    h = ctypes.c_void_p()
    ret = libjpolyd.jprecomp_create(
      ctypes.c_int(self.D), ctypes.c_int(self.n), ctypes.c_int(self.q_pad),
      ctypes.c_int(self.q_vol_requested), ctypes.c_int(self.q_face_requested),
      _ptr_f64(kappa), ctypes.byref(h)
    )
    _check(ret, "jprecomp_create")
    self._handle = h
    self._load_dims()
    self.kappa_res = self._load_kappa_res()

  def __del__(self):
    h = getattr(self, "_handle", None)
    if h:
      libjpolyd.jprecomp_destroy(h)
      self._handle = None

  def _load_dims(self):
    vals = [ctypes.c_int(0) for _ in range(11)]
    ret = libjpolyd.jprecomp_dims(self._handle, *[ctypes.byref(v) for v in vals])
    _check(ret, "jprecomp_dims")
    (self.D, self.n, self.q_vol, self.q_face, self.M, self.m_int,
     self.kf, self.nface, self.nsigma, self.nq_vol, self.nq_face) = [int(v.value) for v in vals]

  def _load_kappa_res(self):
    out = np.empty(self.D + 1, dtype=np.float64)
    ret = libjpolyd.jprecomp_get_kappa_res(self._handle, _ptr_f64(out))
    _check(ret, "jprecomp_get_kappa_res")
    return out

  def dims(self):
    return {
      "D": self.D,
      "n": self.n,
      "q_vol": self.q_vol,
      "q_face": self.q_face,
      "M": self.M,
      "m_int": self.m_int,
      "kf": self.kf,
      "nface": self.nface,
      "nsigma": self.nsigma,
      "nq_vol": self.nq_vol,
      "nq_face": self.nq_face,
    }

  def face_ref_scale(self):
    out = np.empty(self.nface, dtype=np.float64)
    ret = libjpolyd.jprecomp_get_face_ref_scale(self._handle, _ptr_f64(out))
    _check(ret, "jprecomp_get_face_ref_scale")
    return out

  def Lij_ref(self):
    out = np.empty((self.M, self.M, self.D, self.D), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_Lij_ref(self._handle, _ptr_f64(out))
    _check(ret, "jprecomp_get_Lij_ref")
    return out

  def Li_ref(self):
    out = np.empty((self.M, self.M, self.D), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_Li_ref(self._handle, _ptr_f64(out))
    _check(ret, "jprecomp_get_Li_ref")
    return out

  def L0_ref(self):
    out = np.empty((self.M, self.M), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_L0_ref(self._handle, _ptr_f64(out))
    _check(ret, "jprecomp_get_L0_ref")
    return out

  def T_ref(self):
    out = np.empty((self.kf, self.M, self.nsigma, self.nface), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_T_ref(self._handle, _ptr_f64(out))
    _check(ret, "jprecomp_get_T_ref")
    return out

  def Fgrad_ref(self):
    out = np.empty((self.kf, self.M, self.D, self.nsigma, self.nface), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_Fgrad_ref(self._handle, _ptr_f64(out))
    _check(ret, "jprecomp_get_Fgrad_ref")
    return out

  def Mface_ref(self):
    out = np.empty((self.kf, self.kf, self.nface), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_Mface_ref(self._handle, _ptr_f64(out))
    _check(ret, "jprecomp_get_Mface_ref")
    return out

  def volume_quad(self):
    X = np.empty((self.nq_vol, self.D), dtype=np.float64, order="C")
    W = np.empty(self.nq_vol, dtype=np.float64)
    ret = libjpolyd.jprecomp_get_volume_quad(self._handle, _ptr_f64(X), _ptr_f64(W))
    _check(ret, "jprecomp_get_volume_quad")
    return X, W

  def volume_basis(self):
    V = np.empty((self.nq_vol, self.M), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_volume_basis(self._handle, _ptr_f64(V))
    _check(ret, "jprecomp_get_volume_basis")
    return V

  def residual_quad(self):
    X = np.empty((self.nq_vol, self.D), dtype=np.float64, order="C")
    W = np.empty(self.nq_vol, dtype=np.float64)
    ret = libjpolyd.jprecomp_get_residual_quad(
      self._handle, _ptr_f64(X), _ptr_f64(W)
    )
    _check(ret, "jprecomp_get_residual_quad")
    return X, W

  def residual_basis(self):
    V = np.empty((self.nq_vol, self.M), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_residual_basis(self._handle, _ptr_f64(V))
    _check(ret, "jprecomp_get_residual_basis")
    return V

  def residual_quad_basis(self):
    X, W = self.residual_quad()
    V = self.residual_basis()
    return X, W, V

  def face_quad(self):
    Y = np.empty((self.nq_face, max(self.D - 1, 0)), dtype=np.float64, order="C")
    W = np.empty(self.nq_face, dtype=np.float64)

    ret = libjpolyd.jprecomp_get_face_quad(self._handle, _ptr_f64(Y), _ptr_f64(W))
    _check(ret, "jprecomp_get_face_quad")
    return Y, W

  def face_basis(self):
    V = np.empty((self.nq_face, self.kf), dtype=np.float64, order="F")
    ret = libjpolyd.jprecomp_get_face_basis(self._handle, _ptr_f64(V))
    _check(ret, "jprecomp_get_face_basis")
    return V

  def face_quad_basis(self):
    Y, W = self.face_quad()
    V = self.face_basis()
    return Y, W, V
