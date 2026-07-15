#!/usr/bin/env python3
"""Reference precompute equivalence test.

This compares C++ RefSimplexPrecomp reference blocks against the already-tested
Python/C low-level assembly path.  Affine geometry stays in Python.
"""

import argparse
import itertools
import math

import numpy as np

from jbasis import jbasis_build_structures, jbasis_eval_all
from jquad_tprod import jquad_mapped_build_kappa
from jdmat import dmat_build_tprod_natural_pruned
from jkmat import kmat_build_tprod
from jgeom import affine_from_verts
from jperms import face_sigma_array, face_sigma_index, perm_to_lehmer_index, face_vertices
from jlaplace import assemble_L_int
from jtrace import assemble_T_full_common
from jflux import assemble_F_full_common
from jprecomp import RefSimplexPrecomp


def dimPi(D, n):
  return math.comb(n + D, D) if n >= 0 else 0


def dk_natural(D, axis):
  dk = np.zeros(D + 1, dtype=np.float64)
  dk[axis] = 1.0
  dk[D] = 1.0
  return dk


def basis_structs(D, n, kappa):
  alpha, tail, invh = jbasis_build_structures(D, n, kappa)
  return alpha, tail, invh, alpha.shape[0]


def dense(A):
  if hasattr(A, "toarray"):
    return A.toarray()
  return np.asarray(A, dtype=np.float64)


def build_reference_second_partials_old(D, n, q_vol, kappa):
  _, _, _, M = basis_structs(D, n, kappa)
  Lij = np.zeros((M, M, D, D), dtype=np.float64, order="F")
  D1 = []
  k1 = []
  for i in range(D):
    Di = dmat_build_tprod_natural_pruned(D, n, q_vol, kappa, i)
    D1.append(Di)
    k1.append(kappa + dk_natural(D, i))
  for i in range(D):
    for j in range(D):
      Dj = dmat_build_tprod_natural_pruned(D, n, q_vol, k1[i], j)
      k2 = k1[i] + dk_natural(D, j)
      K = kmat_build_tprod(D, n, q_vol, k2, kappa)
      Lij[:, :, i, j] = dense(K @ (Dj @ D1[i]))
  return Lij


def all_sigma_tuples(D):
  out = [None] * math.factorial(D)
  for p in itertools.permutations(range(D)):
    p = np.asarray(p, dtype=np.int32)
    out[int(perm_to_lehmer_index(p))] = p
  return out


def canonical_face_bary(Y):
  if Y.shape[1] == 0:
    return np.ones((Y.shape[0], 1), dtype=np.float64)
  B = np.empty((Y.shape[0], Y.shape[1] + 1), dtype=np.float64)
  B[:, 0] = 1.0 - np.sum(Y, axis=1)
  B[:, 1:] = Y
  return B


def face_points_in_volume_ref(D, face_id, sigma, Y):
  if D == 1:
    X = np.empty((1, 1), dtype=np.float64, order="F")
    X[0, 0] = 1.0 if face_id == 0 else 0.0
    return X
  B_can = canonical_face_bary(Y)
  lam = np.zeros((Y.shape[0], D + 1), dtype=np.float64)
  fv = face_vertices(D, face_id).astype(np.int64)
  for i_local in range(D):
    lam[:, fv[i_local]] = B_can[:, int(sigma[i_local])]
  return np.asfortranarray(lam[:, 1:])


def common_face_basis(D, n, kappa, q_face):
  if D == 1:
    Y = np.zeros((1, 0), dtype=np.float64)
    W = np.ones(1, dtype=np.float64)
    Vt = np.ones((1, 1), dtype=np.float64, order="F")
    return Y, W, Vt, 1
  kappa_face = np.asarray(kappa[:D], dtype=np.float64)
  alpha, tail, invh, kf = basis_structs(D - 1, n, kappa_face)
  Y, W = jquad_mapped_build_kappa(D - 1, q_face, kappa_face)
  Vt = jbasis_eval_all(Y, kappa_face, n, alpha, tail, invh, D - 1)
  return np.asarray(Y), np.asarray(W), np.asfortranarray(Vt), kf


def physical_face_geometry(D, V_phys, global_vids):
  V_phys = np.asarray(V_phys, dtype=np.float64, order="F")
  nface = D + 1
  face_scale = np.empty(nface, dtype=np.float64)
  unit_normal = np.empty((nface, D), dtype=np.float64)
  normal_scaled = np.empty((nface, D), dtype=np.float64)
  for face_id in range(nface):
    fv = face_vertices(D, face_id).astype(np.int64)
    sigma = face_sigma_array(global_vids, face_id).astype(np.int64)
    P_can = np.empty((D, D), dtype=np.float64)
    for i_local in range(D):
      P_can[:, int(sigma[i_local])] = V_phys[:, fv[i_local]]
    if D == 1:
      s = 1.0
      n = np.array([1.0 if face_id == 0 else -1.0], dtype=np.float64)
    else:
      E = P_can[:, 1:] - P_can[:, [0]]
      s = math.sqrt(float(np.linalg.det(E.T @ E)))
      _, _, vh = np.linalg.svd(E.T, full_matrices=True)
      n = vh[-1, :].copy()
      n /= np.linalg.norm(n)
      to_opp = V_phys[:, face_id] - P_can[:, 0]
      if float(np.dot(n, to_opp)) > 0.0:
        n *= -1.0
    face_scale[face_id] = s
    unit_normal[face_id, :] = n
    normal_scaled[face_id, :] = s * n
  return face_scale, unit_normal, normal_scaled


def reference_face_scale(D):
  V = np.zeros((D, D + 1), dtype=np.float64, order="F")
  for j in range(D):
    V[j, j + 1] = 1.0
  g = np.arange(D + 1, dtype=np.int32)
  scale, _, _ = physical_face_geometry(D, V, g)
  return scale


def build_old_face_inputs(D, n, q_face, kappa, global_vids):
  _, _, _, M = basis_structs(D, n, kappa)
  Y, W, Vt, kf = common_face_basis(D, n, kappa, q_face)
  sigmas = all_sigma_tuples(D)
  nsigma = len(sigmas)
  nface = D + 1
  nq = Y.shape[0]

  alpha_v, tail_v, invh_v, _ = basis_structs(D, n, kappa)
  Dcols = []
  k_ranges = []
  structs = []
  for a in range(D):
    kr = kappa + dk_natural(D, a)
    k_ranges.append(kr)
    structs.append(basis_structs(D, n, kr))
    Dcols.append(dmat_build_tprod_natural_pruned(D, n, q_face, kappa, a))

  Vv = np.empty((nq, M, nsigma, nface), dtype=np.float64, order="F")
  dV = np.empty((nq, M, D, nsigma, nface), dtype=np.float64, order="F")
  for f in range(nface):
    for si, sig in enumerate(sigmas):
      Xf = face_points_in_volume_ref(D, f, sig, Y)
      Vv[:, :, si, f] = jbasis_eval_all(Xf, kappa, n, alpha_v, tail_v, invh_v, D)
      for a in range(D):
        alpha_r, tail_r, invh_r, _ = structs[a]
        Vr = jbasis_eval_all(Xf, k_ranges[a], n, alpha_r, tail_r, invh_r, D)
        dV[:, :, a, si, f] = np.asarray(Vr @ Dcols[a])

  face_sigma = np.array([int(face_sigma_index(global_vids, f)) for f in range(nface)], dtype=np.int32)
  return Y, W, Vt, kf, Vv, dV, face_sigma


def make_affine_simplex(D):
  V = np.zeros((D, D + 1), dtype=np.float64, order="F")
  v0 = np.array([0.13 * (i + 1) - 0.07 for i in range(D)], dtype=np.float64)
  B = np.eye(D)
  for i in range(D):
    B[i, i] = 1.1 + 0.17 * i
  for i in range(D):
    for j in range(D):
      if i != j:
        B[i, j] = 0.03 * (i + 1) - 0.02 * (j + 1)
  V[:, 0] = v0
  for j in range(D):
    V[:, j + 1] = v0 + B[:, j]
  return np.asfortranarray(V)


def relerr(A, B):
  return float(np.linalg.norm(A - B) / max(1e-300, np.linalg.norm(B)))


def run_one(D, n, q_pad):
  kappa = np.array([0.7 + 0.19 * i for i in range(D + 1)], dtype=np.float64)
  ref = RefSimplexPrecomp(D, n, kappa, q_pad=q_pad)
  q_vol = ref.q_vol
  q_face = ref.q_face
  print("dims", ref.dims())

  Lij_new = ref.Lij_ref()
  Lij_old = build_reference_second_partials_old(D, n, q_vol, kappa)
  eLij = relerr(Lij_new, Lij_old)

  V_phys = make_affine_simplex(D)
  global_vids = np.array([17 + 11 * i for i in range(D + 1)], dtype=np.int32)
  # scramble ids mildly to force nontrivial sigmas
  if D >= 2:
    global_vids = global_vids[::-1].copy()

  geom = affine_from_verts(V_phys)
  BinvT = geom["BinvT"]
  G = np.asfortranarray(BinvT.T @ BinvT)
  detBabs = float(geom["detBabs"])
  L_old = assemble_L_int(D, G, detBabs, Lij_old, ref.m_int)
  L_new = assemble_L_int(D, G, detBabs, Lij_new, ref.m_int)
  eLint = relerr(L_new, L_old)

  Y, W, Vt, kf, Vv, dV, face_sigma = build_old_face_inputs(D, n, q_face, kappa, global_vids)
  face_scale, unit_normal, normal_scaled = physical_face_geometry(D, V_phys, global_vids)
  T_old = assemble_T_full_common(D, ref.M, kf, Y.shape[0], face_sigma, face_scale, Vt, W, Vv)
  F_old = assemble_F_full_common(D, ref.M, kf, Y.shape[0], face_sigma, normal_scaled, BinvT, Vt, W, dV)

  T_ref = ref.T_ref()
  Fg_ref = ref.Fgrad_ref()
  ref_scale = ref.face_ref_scale()
  T_blocks = []
  F_blocks = []
  for f in range(D + 1):
    si = int(face_sigma[f])
    ratio = face_scale[f] / ref_scale[f]
    T_blocks.append(ratio * T_ref[:, :, si, f])
    eta = BinvT.T @ (ratio * unit_normal[f, :])
    Ff = np.zeros((kf, ref.M), dtype=np.float64)
    for a in range(D):
      Ff += eta[a] * Fg_ref[:, :, a, si, f]
    F_blocks.append(Ff)
  T_new = np.asfortranarray(np.vstack(T_blocks))
  F_new = np.asfortranarray(np.vstack(F_blocks))

  eT = relerr(T_new, T_old)
  eF = relerr(F_new, F_old)
  print(f"[jprecomp equiv] D={D} n={n} q={q_vol} rel_Lij={eLij:.3e} rel_Lint={eLint:.3e} rel_T={eT:.3e} rel_F={eF:.3e}")

  assert eLij < 1e-10, eLij
  assert eLint < 1e-10, eLint
  assert eT < 1e-10, eT
  assert eF < 1e-10, eF


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--D", default="all", help="1,2,3,4,5 or all")
  p.add_argument("--n", type=int, default=6)
  p.add_argument("--q-pad", type=int, default=1)
  args = p.parse_args()
  if args.D == "all":
    Ds = [1, 2, 3, 4, 5]
  else:
    Ds = [int(args.D)]
  for D in Ds:
    run_one(D, args.n, args.q_pad)
  print("All jprecomp equivalence tests passed.")


if __name__ == "__main__":
  main()
