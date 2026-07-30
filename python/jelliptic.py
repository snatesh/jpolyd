import ctypes
import numpy as np

from libjpolyd_loader import libjpolyd

_double_p = ctypes.POINTER(ctypes.c_double)
_int_p = ctypes.POINTER(ctypes.c_int)
_void_pp = ctypes.POINTER(ctypes.c_void_p)

libjpolyd.jelliptic_plan_create.argtypes = [
  ctypes.c_int,
  ctypes.c_int,
  _double_p,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  ctypes.c_int,
  _void_pp,
]
libjpolyd.jelliptic_plan_create.restype = ctypes.c_int

libjpolyd.jelliptic_plan_destroy.argtypes = [ctypes.c_void_p]
libjpolyd.jelliptic_plan_destroy.restype = None

libjpolyd.jelliptic_plan_dims.argtypes = [
  ctypes.c_void_p,
  _int_p,
  _int_p,
  _int_p,
  _int_p,
  _int_p,
  _int_p,
  _int_p,
  _int_p,
]
libjpolyd.jelliptic_plan_dims.restype = ctypes.c_int

libjpolyd.jelliptic_workspace_create.argtypes = [
  ctypes.c_void_p,
  _void_pp,
]
libjpolyd.jelliptic_workspace_create.restype = ctypes.c_int

libjpolyd.jelliptic_workspace_destroy.argtypes = [ctypes.c_void_p]
libjpolyd.jelliptic_workspace_destroy.restype = None

libjpolyd.jelliptic_assemble_L_int.argtypes = [
  ctypes.c_void_p,
  ctypes.c_void_p,
  _double_p,
  ctypes.c_double,
  _double_p,
  _double_p,
  _double_p,
  _double_p,
  _double_p,
  _double_p,
  _double_p,
]
libjpolyd.jelliptic_assemble_L_int.restype = ctypes.c_int


def _ptr_f64(a):
  if a is None:
    return None
  return a.ctypes.data_as(_double_p)


def _check(ret, name):
  if ret == 0:
    return
  if ret == 1:
    raise ValueError(f"{name}: null pointer")
  if ret == 2:
    raise ValueError(f"{name}: invalid or incompatible arguments")
  if ret == 3:
    raise ValueError(f"{name}: unsupported D")
  if ret == 4:
    raise RuntimeError(f"{name}: C++ exception")
  raise RuntimeError(f"{name}: unknown return code {ret}")


class EllipticWorkspace:
  def __init__(self, plan):
    self._plan = plan
    h = ctypes.c_void_p()
    ret = libjpolyd.jelliptic_workspace_create(
      plan._handle,
      ctypes.byref(h),
    )
    _check(ret, "jelliptic_workspace_create")
    self._handle = h

  def close(self):
    h = getattr(self, "_handle", None)
    if h:
      libjpolyd.jelliptic_workspace_destroy(h)
      self._handle = None

  def __del__(self):
    self.close()


class EllipticPlan:
  """Reusable local nondivergence-form elliptic assembly plan."""

  def __init__(self, precomp, p2, p1, p0, assume_symmetric=True):
    self.precomp = precomp
    self.p2 = int(p2)
    self.p1 = int(p1)
    self.p0 = int(p0)

    kappa_res = np.asarray(
      precomp.kappa_res,
      dtype=np.float64,
      order="C",
    )

    h = ctypes.c_void_p()
    ret = libjpolyd.jelliptic_plan_create(
      ctypes.c_int(precomp.D),
      ctypes.c_int(precomp.n),
      _ptr_f64(kappa_res),
      ctypes.c_int(self.p2),
      ctypes.c_int(self.p1),
      ctypes.c_int(self.p0),
      ctypes.c_int(1 if assume_symmetric else 0),
      ctypes.byref(h),
    )
    _check(ret, "jelliptic_plan_create")
    self._handle = h
    self._load_dims()

    if self.D != precomp.D or self.n != precomp.n or self.M != precomp.M:
      self.close()
      raise ValueError("EllipticPlan: precompute/plan dimension mismatch")

    # Binding-only copies. Direct C++ Leaf use consumes RefSimplexPrecomp arrays
    # without these copies.
    self._Lij_ref = np.asfortranarray(precomp.Lij_ref())
    self._Li_ref = np.asfortranarray(precomp.Li_ref())
    self._L0_ref = np.asfortranarray(precomp.L0_ref())
    self._default_workspace = EllipticWorkspace(self)

  def _load_dims(self):
    vals = [ctypes.c_int(0) for _ in range(8)]
    ret = libjpolyd.jelliptic_plan_dims(
      self._handle,
      *[ctypes.byref(v) for v in vals],
    )
    _check(ret, "jelliptic_plan_dims")
    (
      self.D,
      self.n,
      self.M,
      self.m2,
      self.m1,
      self.Mp2,
      self.Mp1,
      self.Mp0,
    ) = [int(v.value) for v in vals]

  def create_workspace(self):
    return EllipticWorkspace(self)

  def close(self):
    default = getattr(self, "_default_workspace", None)
    if default is not None:
      default.close()
      self._default_workspace = None

    h = getattr(self, "_handle", None)
    if h:
      libjpolyd.jelliptic_plan_destroy(h)
      self._handle = None

  def __del__(self):
    self.close()

  def assemble_L_int(
    self,
    BinvT,
    detBabs,
    A=None,
    b=None,
    c=None,
    workspace=None,
    out=None,
  ):
    BinvT = np.asfortranarray(BinvT, dtype=np.float64)
    if BinvT.shape != (self.D, self.D):
      raise ValueError(f"BinvT must have shape ({self.D}, {self.D})")

    if self.p2 >= 0:
      A = np.asarray(A, dtype=np.float64, order="C")
      if A.shape != (self.D, self.D, self.Mp2):
        raise ValueError(
          f"A must have shape ({self.D}, {self.D}, {self.Mp2})"
        )
    else:
      A = None

    if self.p1 >= 0:
      b = np.asarray(b, dtype=np.float64, order="C")
      if b.shape != (self.D, self.Mp1):
        raise ValueError(f"b must have shape ({self.D}, {self.Mp1})")
    else:
      b = None

    if self.p0 >= 0:
      c = np.asarray(c, dtype=np.float64, order="C")
      if c.shape != (self.Mp0,):
        raise ValueError(f"c must have shape ({self.Mp0},)")
    else:
      c = None

    if out is None:
      out = np.empty((self.m2, self.M), dtype=np.float64, order="F")
    else:
      if not isinstance(out, np.ndarray):
        raise TypeError("out must be a numpy.ndarray")
      if out.dtype != np.float64 or out.shape != (self.m2, self.M):
        raise ValueError(f"out must have shape ({self.m2}, {self.M}) and dtype float64")
      if not out.flags.f_contiguous:
        raise ValueError("out must be Fortran contiguous")

    if workspace is None:
      workspace = self._default_workspace
    if workspace._plan is not self:
      raise ValueError("workspace belongs to a different EllipticPlan")

    ret = libjpolyd.jelliptic_assemble_L_int(
      self._handle,
      workspace._handle,
      _ptr_f64(BinvT),
      ctypes.c_double(float(detBabs)),
      _ptr_f64(self._Lij_ref),
      _ptr_f64(self._Li_ref),
      _ptr_f64(self._L0_ref),
      _ptr_f64(A),
      _ptr_f64(b),
      _ptr_f64(c),
      _ptr_f64(out),
    )
    _check(ret, "jelliptic_assemble_L_int")
    return out
