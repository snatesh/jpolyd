import ctypes
import math
import numpy as np

from libjpolyd_loader import libjpolyd


_double_p = ctypes.POINTER(ctypes.c_double)
_int_p = ctypes.POINTER(ctypes.c_int)

libjpolyd.jtrace_assemble_T_full_common.argtypes = [
  ctypes.c_int,    # D
  ctypes.c_int,    # M
  ctypes.c_int,    # kf
  ctypes.c_int,    # nq
  _int_p,          # face_sigma_index, length D+1
  _double_p,       # face_scale, length D+1
  _double_p,       # Vt_common, shape (nq,kf), F order
  _double_p,       # wS_hat_common, length nq
  _double_p,       # Vv_sigma_face, packed shape (nq,M,nsigma,nface), F order
  _double_p,       # T_full_out, shape ((D+1)*kf,M), F order
]
libjpolyd.jtrace_assemble_T_full_common.restype = ctypes.c_int

libjpolyd.jtrace_assemble_T_full_facepacked.argtypes = [
  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
  _int_p, _double_p, _double_p, _double_p, _double_p, _double_p,
]
libjpolyd.jtrace_assemble_T_full_facepacked.restype = ctypes.c_int


def _ptr_f64(a):
  return a.ctypes.data_as(_double_p)


def _ptr_i32(a):
  return a.ctypes.data_as(_int_p)


def _check_ret(ret, name):
  if ret == 0:
    return
  if ret == 1:
    raise ValueError(f"{name}: null pointer passed to C API")
  if ret == 2:
    raise ValueError(f"{name}: invalid dimensions")
  if ret == 3:
    raise ValueError(f"{name}: unsupported D")
  if ret == 4:
    raise ValueError(f"{name}: invalid face sigma index")
  raise RuntimeError(f"{name}: failed with unknown return code {ret}")


def num_face_perms(D):
  return math.factorial(int(D))


def assemble_T_full_common(D, M, kf, nq, face_sigma_index, face_scale,
                           Vt_common, wS_hat_common, Vv_sigma_face):
  """
  Assemble stacked T_full using one common face basis for all faces.

  Parameters
  ----------
  D : int
      Simplex dimension. Number of faces is D+1.
  M : int
      Volume basis dimension.
  kf : int
      Face basis dimension.
  nq : int
      Face quadrature point count.
  face_sigma_index : array_like, shape (D+1,), int32
      Orientation index per face.
  face_scale : array_like, shape (D+1,), float64
      Surface scaling per face.
  Vt_common : array_like, shape (nq,kf), Fortran-order preferred
      Common canonical face basis evaluations.
  wS_hat_common : array_like, shape (nq,), float64
      Common canonical face quadrature weights.
  Vv_sigma_face : array_like
      Either shape (nq,M,nsigma,D+1) in Fortran order, or a 1D packed buffer.
      Block (face,sigma) is Vv_sigma_face[:,:,sigma,face].

  Returns
  -------
  T : ndarray, shape ((D+1)*kf, M), Fortran order
      Stacked trace matrix. Face f occupies rows f*kf:(f+1)*kf.
  """
  D = int(D)
  M = int(M)
  kf = int(kf)
  nq = int(nq)
  nface = D + 1
  nsigma = num_face_perms(D)

  face_sigma_index = np.asarray(face_sigma_index, dtype=np.int32, order="C")
  if face_sigma_index.shape != (nface,):
    raise ValueError(f"face_sigma_index must have shape ({nface},)")

  face_scale = np.asarray(face_scale, dtype=np.float64, order="C")
  if face_scale.shape != (nface,):
    raise ValueError(f"face_scale must have shape ({nface},)")

  Vt_common = np.asarray(Vt_common, dtype=np.float64, order="F")
  if Vt_common.shape != (nq, kf):
    raise ValueError(f"Vt_common must have shape ({nq}, {kf})")

  wS_hat_common = np.asarray(wS_hat_common, dtype=np.float64, order="C")
  if wS_hat_common.shape != (nq,):
    raise ValueError(f"wS_hat_common must have shape ({nq},)")

  Vv_sigma_face = np.asarray(Vv_sigma_face, dtype=np.float64, order="F")
  if Vv_sigma_face.ndim == 4:
    if Vv_sigma_face.shape != (nq, M, nsigma, nface):
      raise ValueError(
        f"Vv_sigma_face must have shape ({nq}, {M}, {nsigma}, {nface})"
      )
    Vv_packed = Vv_sigma_face
  elif Vv_sigma_face.ndim == 1:
    if Vv_sigma_face.size != nq * M * nsigma * nface:
      raise ValueError("1D Vv_sigma_face has wrong size")
    Vv_packed = np.asfortranarray(
      Vv_sigma_face.reshape((nq, M, nsigma, nface), order="F")
    )
  else:
    raise ValueError("Vv_sigma_face must be 4D or 1D")

  T = np.zeros((nface * kf, M), dtype=np.float64, order="F")

  ret = libjpolyd.jtrace_assemble_T_full_common(
    ctypes.c_int(D), ctypes.c_int(M), ctypes.c_int(kf), ctypes.c_int(nq),
    _ptr_i32(face_sigma_index),
    _ptr_f64(face_scale),
    _ptr_f64(Vt_common),
    _ptr_f64(wS_hat_common),
    _ptr_f64(Vv_packed),
    _ptr_f64(T),
  )
  _check_ret(ret, "jtrace_assemble_T_full_common")
  return T


def assemble_T_full_facepacked(D, M, kf, nq, face_sigma_index, face_scale,
                               Vt_face, wS_hat_face, Vv_sigma_face):
  """
  Compatibility wrapper with face-packed face basis/weights.

  Vt_face shape is (nq,kf,D+1), Fortran order.
  wS_hat_face shape is (nq,D+1), Fortran order.
  Vv_sigma_face shape is (nq,M,nsigma,D+1), Fortran order.
  """
  D = int(D)
  M = int(M)
  kf = int(kf)
  nq = int(nq)
  nface = D + 1
  nsigma = num_face_perms(D)

  face_sigma_index = np.asarray(face_sigma_index, dtype=np.int32, order="C")
  face_scale = np.asarray(face_scale, dtype=np.float64, order="C")
  Vt_face = np.asarray(Vt_face, dtype=np.float64, order="F")
  wS_hat_face = np.asarray(wS_hat_face, dtype=np.float64, order="F")
  Vv_sigma_face = np.asarray(Vv_sigma_face, dtype=np.float64, order="F")

  if face_sigma_index.shape != (nface,):
    raise ValueError(f"face_sigma_index must have shape ({nface},)")
  if face_scale.shape != (nface,):
    raise ValueError(f"face_scale must have shape ({nface},)")
  if Vt_face.shape != (nq, kf, nface):
    raise ValueError(f"Vt_face must have shape ({nq}, {kf}, {nface})")
  if wS_hat_face.shape != (nq, nface):
    raise ValueError(f"wS_hat_face must have shape ({nq}, {nface})")
  if Vv_sigma_face.shape != (nq, M, nsigma, nface):
    raise ValueError(
      f"Vv_sigma_face must have shape ({nq}, {M}, {nsigma}, {nface})"
    )

  T = np.zeros((nface * kf, M), dtype=np.float64, order="F")

  ret = libjpolyd.jtrace_assemble_T_full_facepacked(
    ctypes.c_int(D), ctypes.c_int(M), ctypes.c_int(kf), ctypes.c_int(nq),
    _ptr_i32(face_sigma_index),
    _ptr_f64(face_scale),
    _ptr_f64(Vt_face),
    _ptr_f64(wS_hat_face),
    _ptr_f64(Vv_sigma_face),
    _ptr_f64(T),
  )
  _check_ret(ret, "jtrace_assemble_T_full_facepacked")
  return T
