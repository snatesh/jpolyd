import math
import numpy as np

from jtrace import assemble_T_full_common, assemble_T_full_facepacked


def reference_T_common(D, M, kf, nq, face_sigma_index, face_scale,
                       Vt_common, wS_hat_common, Vv_sigma_face):
  nface = D + 1
  T = np.zeros((nface * kf, M), dtype=np.float64, order="F")
  for f in range(nface):
    sig = int(face_sigma_index[f])
    Vv = Vv_sigma_face[:, :, sig, f]
    weighted = (wS_hat_common * face_scale[f])[:, None] * Vv
    T[f*kf:(f+1)*kf, :] = Vt_common.T @ weighted
  return T


def reference_T_facepacked(D, M, kf, nq, face_sigma_index, face_scale,
                           Vt_face, wS_hat_face, Vv_sigma_face):
  nface = D + 1
  T = np.zeros((nface * kf, M), dtype=np.float64, order="F")
  for f in range(nface):
    sig = int(face_sigma_index[f])
    Vv = Vv_sigma_face[:, :, sig, f]
    weighted = (wS_hat_face[:, f] * face_scale[f])[:, None] * Vv
    T[f*kf:(f+1)*kf, :] = Vt_face[:, :, f].T @ weighted
  return T


def run_one(D, seed):
  rng = np.random.default_rng(seed)
  nface = D + 1
  nsigma = math.factorial(D)

  M = 7 + D
  kf = 7 + D
  nq = 9 + D

  face_sigma_index = rng.integers(0, nsigma, size=nface, dtype=np.int32)
  face_scale = np.ascontiguousarray(0.5 + rng.random(nface))

  Vt_common = np.asfortranarray(rng.normal(size=(nq, kf)))
  wS_common = np.ascontiguousarray(0.25 + rng.random(nq))
  Vv = np.asfortranarray(rng.normal(size=(nq, M, nsigma, nface)))

  T_c = assemble_T_full_common(D, M, kf, nq, face_sigma_index, face_scale,
                               Vt_common, wS_common, Vv)
  T_ref = reference_T_common(D, M, kf, nq, face_sigma_index, face_scale,
                             Vt_common, wS_common, Vv)

  np.testing.assert_allclose(T_c, T_ref, rtol=5e-13, atol=5e-13)

  Vt_face = np.asfortranarray(rng.normal(size=(nq, kf, nface)))
  wS_face = np.asfortranarray(0.25 + rng.random(size=(nq, nface)))

  T_c2 = assemble_T_full_facepacked(D, M, kf, nq, face_sigma_index, face_scale,
                                    Vt_face, wS_face, Vv)
  T_ref2 = reference_T_facepacked(D, M, kf, nq, face_sigma_index, face_scale,
                                  Vt_face, wS_face, Vv)

  np.testing.assert_allclose(T_c2, T_ref2, rtol=5e-13, atol=5e-13)

  print(
    f"D={D}: nface={nface}, nsigma={nsigma}, M={M}, kf={kf}, nq={nq}, "
    f"maxerr_common={np.max(np.abs(T_c - T_ref)):.3e}, "
    f"maxerr_facepacked={np.max(np.abs(T_c2 - T_ref2)):.3e}"
  )


def run_invalid_sigma_test():
  D = 3
  M, kf, nq = 5, 4, 7
  nface = D + 1
  nsigma = math.factorial(D)

  face_sigma_index = np.zeros(nface, dtype=np.int32)
  face_sigma_index[0] = nsigma
  face_scale = np.ones(nface)
  Vt = np.asfortranarray(np.ones((nq, kf)))
  wS = np.ones(nq)
  Vv = np.asfortranarray(np.ones((nq, M, nsigma, nface)))

  try:
    assemble_T_full_common(D, M, kf, nq, face_sigma_index, face_scale, Vt, wS, Vv)
  except ValueError as exc:
    print("invalid sigma correctly rejected:", exc)
    return

  raise AssertionError("Expected invalid sigma to be rejected")


if __name__ == "__main__":
  for D in range(1, 6):
    run_one(D, seed=1000 + D)
  run_invalid_sigma_test()
  print("jtrace ctypes tests passed.")
