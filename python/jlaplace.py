import ctypes
import numpy as np

from libjpolyd_loader import libjpolyd


_double_p = ctypes.POINTER(ctypes.c_double)

libjpolyd.jlaplace_assemble_L_int.argtypes = [
  ctypes.c_int,     # D
  ctypes.c_int,     # M
  ctypes.c_int,     # m_int
  _double_p,        # G, shape (D,D), F order
  ctypes.c_double,  # detBabs
  _double_p,        # Lij_ref, shape (M,M,D,D), F order
  _double_p,        # L_int_out, shape (m_int,M), F order
]
libjpolyd.jlaplace_assemble_L_int.restype = ctypes.c_int


def _ptr_f64(a):
  return a.ctypes.data_as(_double_p)


def _check_ret(ret, name):
  if ret == 0:
    return
  if ret == 1:
    raise ValueError(f"{name}: null pointer passed to C API")
  if ret == 2:
    raise ValueError(f"{name}: invalid dimensions")
  if ret == 3:
    raise ValueError(f"{name}: unsupported D")
  raise RuntimeError(f"{name}: failed with unknown return code {ret}")


def assemble_L_int(D, G, detBabs, Lij_ref, m_int):
  """
  Assemble the affine interior Laplacian matrix.

  Parameters
  ----------
  D : int
      Simplex dimension.
  G : array_like, shape (D,D)
      Affine metric G = B^{-1} B^{-T}. Fortran-order preferred.
  detBabs : float
      Absolute determinant |det B| of the affine map.
  Lij_ref : array_like, shape (M,M,D,D)
      Reference promoted second-derivative blocks. Fortran-order preferred.
      Lij_ref[:,:,i,j] maps Pi_n^D coefficients to promoted coefficients of
      d^2/dxhat_i dxhat_j in the target/reference parameter convention.
  m_int : int
      Number of interior rows, usually dim Pi_{n-2}^D.

  Returns
  -------
  L_int : ndarray, shape (m_int,M), Fortran order
      L_int = |detB| * (sum_ij G_ij Lij_ref[:,:,i,j])[:m_int,:].
  """
  D = int(D)
  m_int = int(m_int)

  G = np.asarray(G, dtype=np.float64, order="F")
  if G.shape != (D, D):
    raise ValueError(f"G must have shape ({D}, {D})")

  Lij_ref = np.asarray(Lij_ref, dtype=np.float64, order="F")
  if Lij_ref.ndim != 4:
    raise ValueError("Lij_ref must have shape (M,M,D,D)")
  M = int(Lij_ref.shape[0])
  if Lij_ref.shape != (M, M, D, D):
    raise ValueError(f"Lij_ref must have shape (M, M, {D}, {D})")
  if m_int <= 0 or m_int > M:
    raise ValueError("m_int must satisfy 0 < m_int <= M")

  L_int = np.zeros((m_int, M), dtype=np.float64, order="F")

  ret = libjpolyd.jlaplace_assemble_L_int(
    ctypes.c_int(D),
    ctypes.c_int(M),
    ctypes.c_int(m_int),
    _ptr_f64(G),
    ctypes.c_double(float(detBabs)),
    _ptr_f64(Lij_ref),
    _ptr_f64(L_int),
  )
  _check_ret(ret, "jlaplace_assemble_L_int")
  return L_int
