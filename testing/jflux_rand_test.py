import math
import numpy as np

from jflux import assemble_F_full_common, assemble_F_full_facepacked


def num_face_perms(D):
  return math.factorial(int(D))


def reference_common(D, M, kf, nq, face_sigma_index, normal_scaled, BinvT,
                     Vt_common, W, dV):
  nface = D + 1
  F = np.zeros((nface * kf, M), dtype=np.float64, order="F")
  for f in range(nface):
    sig = int(face_sigma_index[f])
    beta = normal_scaled[f, :] @ BinvT
    ndot = np.tensordot(dV[:, :, :, sig, f], beta, axes=([2], [0]))
    F[f*kf:(f+1)*kf, :] = Vt_common.T @ (W[:, None] * ndot)
  return F


def reference_facepacked(D, M, kf, nq, face_sigma_index, normal_scaled, BinvT,
                         Vt_face, W_face, dV):
  nface = D + 1
  F = np.zeros((nface * kf, M), dtype=np.float64, order="F")
  for f in range(nface):
    sig = int(face_sigma_index[f])
    beta = normal_scaled[f, :] @ BinvT
    ndot = np.tensordot(dV[:, :, :, sig, f], beta, axes=([2], [0]))
    F[f*kf:(f+1)*kf, :] = Vt_face[:, :, f].T @ (W_face[:, [f]] * ndot)
  return F


def run_one(D, seed=100):
  rng = np.random.default_rng(seed + D)
  nface = D + 1
  nsigma = num_face_perms(D)
  nq = 5 + D
  M = 4 + 3 * D
  kf = 3 + D

  face_sigma_index = rng.integers(0, nsigma, size=nface, dtype=np.int32)
  normal_scaled = np.ascontiguousarray(rng.normal(size=(nface, D)))

  A = rng.normal(size=(D, D))
  BinvT = np.asfortranarray(A + D * np.eye(D))

  Vt_common = np.asfortranarray(rng.normal(size=(nq, kf)))
  W = np.ascontiguousarray(0.2 + rng.random(nq))
  dV = np.asfortranarray(rng.normal(size=(nq, M, D, nsigma, nface)))

  Fc = assemble_F_full_common(
    D, M, kf, nq,
    face_sigma_index,
    normal_scaled,
    BinvT,
    Vt_common,
    W,
    dV,
  )
  Fr = reference_common(D, M, kf, nq, face_sigma_index, normal_scaled,
                        BinvT, Vt_common, W, dV)
  np.testing.assert_allclose(Fc, Fr, rtol=2e-14, atol=2e-14)

  Vt_face = np.asfortranarray(rng.normal(size=(nq, kf, nface)))
  W_face = np.asfortranarray(0.2 + rng.random(size=(nq, nface)))
  Fcp = assemble_F_full_facepacked(
    D, M, kf, nq,
    face_sigma_index,
    normal_scaled,
    BinvT,
    Vt_face,
    W_face,
    dV,
  )
  Frp = reference_facepacked(D, M, kf, nq, face_sigma_index, normal_scaled,
                             BinvT, Vt_face, W_face, dV)
  np.testing.assert_allclose(Fcp, Frp, rtol=2e-14, atol=2e-14)

  bad = face_sigma_index.copy()
  bad[0] = nsigma
  try:
    assemble_F_full_common(D, M, kf, nq, bad, normal_scaled, BinvT, Vt_common, W, dV)
  except ValueError:
    pass
  else:
    raise AssertionError("invalid sigma index was not rejected")

  print(f"[jflux algebra] D={D} M={M} kf={kf} nq={nq} nsigma={nsigma} passed")


def run_smoke():
  for D in range(1, 6):
    run_one(D)
  print("jflux algebra/packing smoke test passed.")


if __name__ == "__main__":
  run_smoke()
