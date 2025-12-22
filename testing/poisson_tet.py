import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg
import sparseqr

import sympy as sp
from math import comb
import time

from jdmat import dmat_build_tprod_natural_pruned
from jkmat import kmat_build_tprod
from jquad_tprod import jquad_mapped_build_kappa
from jbasis import (
  jbasis_build_structures,
  jbasis_eval_all,
  jbasis_eval_all_with_grad,
)

def dim_Pi(D, n):
  return comb(n + D, D)

def dk_natural(D, axis):
  dk = np.zeros(D + 1, dtype=np.float64)
  dk[axis] = 1.0
  dk[D] = 1.0
  return dk

def nnz_stats(A, thr_list=(1e-10, 1e-12, 1e-14, 1e-16)):
  A = np.asarray(A, dtype=np.float64)
  fro = np.linalg.norm(A, ord="fro")
  s = fro if fro != 0.0 else 1.0 
  stats = []
  for thr in thr_list:
    nnz = int(np.count_nonzero((np.abs(A) / s) > thr))
    stats.append((thr, nnz, nnz / A.size))
  return fro, stats
#fro, stats = nnz_stats(self.T_full)
#print("[T_full] ||.||_F =", fro)
#for thr, nnz, frac in stats:
#  print(f"[T_full] nnz(frac) for abs/||.||_F > {thr:g}: {nnz} ({frac:.6e})")


def tet_affine_from_verts(V):
  V = np.asarray(V, dtype=np.float64)
  if V.shape != (4, 3):
    raise ValueError("V must have shape (4,3)")

  v0 = V[0, :]
  J = np.stack([V[1, :] - v0,
                V[2, :] - v0,
                V[3, :] - v0], axis=1)
  b = v0.copy()

  detJ = float(np.linalg.det(J))
  detJabs = abs(detJ)

  Jinv = np.linalg.inv(J)
  JinvT = Jinv.T
  G = Jinv @ JinvT

  return J, b, detJ, detJabs, Jinv, JinvT, G

def tet_face_outward_sign(V, face_id):
  """
  V: (4,3) tet vertices in leaf's local order.
  face_id: 0..3, LOCAL_FACE_TRIS gives the face vertices.
  Returns s in {+1,-1} such that s * n_face_local is outward.
  """
  tri = LOCAL_FACE_TRIS[int(face_id)]
  i0, i1, i2 = tri
  # opposite vertex index
  opp = [0,1,2,3]
  opp.remove(i0); opp.remove(i1); opp.remove(i2)
  i3 = opp[0]

  p0 = V[i0]; p1 = V[i1]; p2 = V[i2]; p3 = V[i3]

  # local face normal (not unit), depends on (i0,i1,i2) ordering
  n = np.cross(p1 - p0, p2 - p0)

  # vector from face towards opposite vertex
  v = p3 - p0

  # If n points toward the opposite vertex, it's inward (needs flip).
  # Outward means pointing away from interior, i.e. opposite sign.
  # So outward if dot(n, v) < 0.
  return 1.0 if float(np.dot(n, v)) < 0.0 else -1.0

def map_ref_to_phys(Xhat, J, b):
  return (Xhat @ J.T) + b[None, :]

def interior_qr_factor(L, rtol=1e-14):
  L = np.asarray(L, dtype=np.float64)
  m, M = L.shape
  
  Q, R, piv = scipy.linalg.qr(L, pivoting=True, mode="economic")
  piv = np.asarray(piv, dtype=np.int64)
    
  d = np.abs(np.diag(R))
  if d.size == 0:
    raise RuntimeError("empty diag(R) in interior QR")
    
  dmax = d.max()
  if dmax == 0.0:
    raise RuntimeError("interior QR diag is all-zero")
  
  rnk = int(np.sum(d > rtol * dmax))
  if rnk < m:
    raise RuntimeError(f"interior block appears rank-deficient: rank={rnk}, m={m}")
    
  R11 = R[:m, :m]
  R12 = R[:m, m:] 
  rnull = M - m
  
  if rnull == 0:
    N = np.zeros((M, 0), dtype=np.float64)
  else:
    B = -scipy.linalg.solve_triangular(R11, R12, lower=False, check_finite=False)
    Nperm = np.zeros((M, rnull), dtype=np.float64)
    Nperm[:m, :] = B
    Nperm[m:, :] = np.eye(rnull, dtype=np.float64)
    N = np.zeros_like(Nperm)
    N[piv, :] = Nperm

  return {"Q": Q, "R11": R11, "piv": piv, "m": m, "M": M, "N": N}

def all_S3_perms():
  return [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
  ]

LOCAL_FACE_TRIS = [
  (1, 2, 3),
  (0, 2, 3),
  (0, 1, 3),
  (0, 1, 2),
]

def tri_coords_perm(u, v, sigma):
  u = np.asarray(u, dtype=np.float64)
  v = np.asarray(v, dtype=np.float64)
  l0 = 1.0 - u - v
  l1 = u
  l2 = v
  l = [l0, l1, l2]
  return l[sigma[1]], l[sigma[2]]


def face_sigma_local_to_canonical(local_gtriple_ordered, canonical_gtriple):
  canon = list(canonical_gtriple)
  loc = list(local_gtriple_ordered)
  sigma = []
  for i in range(3):
    sigma.append(canon.index(loc[i]))
  return tuple(sigma)


def build_face_sigma_list_for_tet(tet_gverts):
  sigmas = []
  for face_id, tri in enumerate(LOCAL_FACE_TRIS):
    local_gtriple = (tet_gverts[tri[0]], tet_gverts[tri[1]], tet_gverts[tri[2]])
    canon = tuple(sorted(local_gtriple))
    sigma = face_sigma_local_to_canonical(local_gtriple, canon)
    sigmas.append(sigma)
  return sigmas


def kappa_face_from_kappa_src(kappa_src, face_id):
  if face_id == 0:
    return np.array([kappa_src[1], kappa_src[2], kappa_src[3]], dtype=np.float64)
  if face_id == 1:
    return np.array([kappa_src[0], kappa_src[2], kappa_src[3]], dtype=np.float64)
  if face_id == 2:
    return np.array([kappa_src[0], kappa_src[1], kappa_src[3]], dtype=np.float64)
  if face_id == 3:
    return np.array([kappa_src[0], kappa_src[1], kappa_src[2]], dtype=np.float64)
  raise ValueError("bad face_id")


def face_map_and_geom(face_id, u, v):
  if face_id == 0:
    Xf = np.stack([1.0 - u - v, u, v], axis=1)
    area_scale = np.sqrt(3.0)
    n_hat = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
  elif face_id == 1:
    Xf = np.stack([0*u, u, v], axis=1)
    area_scale = 1.0
    n_hat = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
  elif face_id == 2:
    Xf = np.stack([u, 0*u, v], axis=1)
    area_scale = 1.0
    n_hat = np.array([0.0, -1.0, 0.0], dtype=np.float64)
  elif face_id == 3:
    Xf = np.stack([u, v, 0*u], axis=1)
    area_scale = 1.0
    n_hat = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  else:
    raise ValueError("bad face_id")
  return Xf, area_scale, n_hat

def ref_face_normal_norm(face_id):
  # reference tet vertices
  V = np.array([[0.0,0.0,0.0],
                [1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,1.0]])
  tri = LOCAL_FACE_TRIS[int(face_id)]
  p0, p1, p2 = V[tri[0]], V[tri[1]], V[tri[2]]
  n = np.cross(p1 - p0, p2 - p0)
  return float(np.linalg.norm(n))

def _dense_to_csc_pruned(A, rel=1e-14, abs_tol=0.0):
  """
  Convert dense A to CSC, pruning entries with |a_ij| <= max(abs_tol, rel*||A||_F).
  """
  A = np.asarray(A)
  fro = float(np.linalg.norm(A))
  thr = max(float(abs_tol), float(rel) * max(1.0, fro))
  if thr > 0.0:
    A = A.copy()
    A[np.abs(A) <= thr] = 0.0
  return sps.csc_matrix(A)

class RefTetPrecomp:
  def __init__(self, n, q_vol, q_face, kappa_src):
    self.D = 3
    self.n = int(n)
    self.q_vol = int(q_vol)
    self.q_face = int(q_face)
    self.kappa_src = np.asarray(kappa_src, dtype=np.float64)
    self.face_sigma = [(0,1,2), (0,1,2), (0,1,2), (0,1,2)] 
    self.face_nref_norm = np.array([ref_face_normal_norm(f) for f in range(4)], dtype=np.float64)
    v0 = np.array([0.0,0.0,0.0])
    v1 = np.array([1.0,0.0,0.0])
    v2 = np.array([0.0,1.0,0.0])
    v3 = np.array([0.0,0.0,1.0])
    self.V_phys = np.stack([v0, v1, v2, v3], axis=0)
    J, b, detJ, detJabs, Jinv, JinvT, G = tet_affine_from_verts(self.V_phys)
    self.J = J
    self.b = b
    self.detJ = detJ
    self.detJabs = detJabs
    self.JinvT = JinvT
    self.G = G
    if self.n < 2:
      raise ValueError("need n>=2")

    self.M = dim_Pi(3, self.n)
    self.m = dim_Pi(3, self.n - 2)
    self.k = (self.n + 1) * (self.n + 1)
    self.kf = dim_Pi(2, self.n)

    self.kappa_lap = self.kappa_src + 2.0 * np.ones(4, dtype=np.float64)

    # volume quad + basis evals (src)
    Xhat_src, w_src = jquad_mapped_build_kappa(3, self.q_vol, self.kappa_src)
    alpha_src, tail_src, invh_src = jbasis_build_structures(3, self.n, self.kappa_src)
    V_src = jbasis_eval_all(Xhat_src, self.kappa_src, self.n, alpha_src, tail_src, invh_src, 3)

    # volume quad + basis evals (lap)
    Xhat_lap, w_lap = jquad_mapped_build_kappa(3, self.q_vol, self.kappa_lap)
    alpha_lap, tail_lap, invh_lap = jbasis_build_structures(3, self.n, self.kappa_lap)
    V_lap = jbasis_eval_all(Xhat_lap, self.kappa_lap, self.n, alpha_lap, tail_lap, invh_lap, 3)

    self.alpha_src = alpha_src
    self.tail_src = tail_src
    self.invh_src = invh_src

    self.Xhat_vol_src = Xhat_src
    self.w_vol_src = w_src
    self.V_vol_src = V_src

    self.Xhat_vol_lap = Xhat_lap
    self.w_vol_lap = w_lap
    self.V_vol_lap = V_lap

    # faces
    self.face = []
    perms = all_S3_perms()
    T_sigma_ref = {}         # dense (optional)
    T_sigma_ref_csc = {}     # pruned sparse CSC (recommended)
    F_sigma_ref = {}
    F_sigma_ref_csc = {}
    for face_id in range(4):
      kappa_tri = kappa_face_from_kappa_src(self.kappa_src, face_id)

      Xt, wt = jquad_mapped_build_kappa(2, self.q_face, kappa_tri)
      u = Xt[:, 0]
      v = Xt[:, 1]

      _, area_scale, n_hat = face_map_and_geom(face_id, u, v)
      wS_hat = wt * area_scale

      alpha_tri, tail_tri, invh_tri = jbasis_build_structures(2, self.n, kappa_tri)
      Vt = jbasis_eval_all(Xt, kappa_tri, self.n, alpha_tri, tail_tri, invh_tri, 2)

      Vv_sigma = {}
      dVv_hat_sigma = {}
      Xf_hat_sigma = {}
      T_sigma_ref = {}
      T_sigma_ref_csc = {}
      F_sigma_ref = {}
      F_sigma_ref_csc = {}
      for sigma in perms:
        u_loc, v_loc = tri_coords_perm(u, v, sigma)
        Xf_hat, _, _ = face_map_and_geom(face_id, u_loc, v_loc)
        Vv, dVv_hat = jbasis_eval_all_with_grad(
          Xf_hat, self.kappa_src, self.n, alpha_src, tail_src, invh_src, 3
        )
        Xf_hat_sigma[sigma] = Xf_hat
        Vv_sigma[sigma] = Vv
        dVv_hat_sigma[sigma] = dVv_hat

        # -----------------------------
        # Build the reference trace block for this face+sigma:
        #   T = Vt^T * diag(wS_hat) * Vv
        # where wS_hat already includes reference-face area_scale.
        # -----------------------------

        n_tilde = self.JinvT @ n_hat
        n_tilde_norm = float(np.linalg.norm(n_tilde))
        wS_phys = wS_hat * (self.detJabs * n_tilde_norm)
        T = Vt.T @ (wS_phys[:, None] * Vv)
        T_sigma_ref[sigma] = T
        T_sigma_ref_csc[sigma] = _dense_to_csc_pruned(T, rel=1e-14, abs_tol=0.0)

        wF_phys = wS_hat * self.detJabs
        dVv_x = np.einsum("ab,qmb->qma", self.JinvT, dVv_hat)
        ndot = (n_tilde[0] * dVv_x[:, :, 0] +
                n_tilde[1] * dVv_x[:, :, 1] +
                n_tilde[2] * dVv_x[:, :, 2])
        F = Vt.T @ (wF_phys[:, None] * ndot)
        F_sigma_ref[sigma] = F
        F_sigma_ref_csc[sigma] = _dense_to_csc_pruned(F, rel=1e-14, abs_tol=0.0)
        
      self.face.append({
        "face_id": face_id,
        "Xt": Xt,
        "wt": wt,
        "wS_hat": wS_hat,
        "Vt": Vt,
        "n_hat": n_hat,
        "Xf_hat_sigma": Xf_hat_sigma,
        "Vv_sigma": Vv_sigma,
        "dVv_hat_sigma": dVv_hat_sigma,
        "T_sigma_ref": T_sigma_ref,
        "T_sigma_ref_csc": T_sigma_ref_csc,
        "F_sigma_ref": F_sigma_ref,
        "F_sigma_ref_csc": F_sigma_ref_csc
      })


    for f in range(4):
      d = self.face[f]["T_sigma_ref_csc"]
      nnz_list = [d[s].nnz for s in perms]
      #print("face", f, "nnz min/max over sigma:", min(nnz_list), max(nnz_list))
    
      # also compare Frobenius norms (should vary mildly, not explode)
      fn_list = [sps.linalg.norm(d[s]) for s in perms]
      #print("face", f, "||T||_F min/max over sigma:", min(fn_list), max(fn_list))

    self.Lij, self.Lij_csc = self._precompute_promoted_second_partials()
    # laplacian on ref
    self.L_ref = self.Lij[0][0] + self.Lij[1][1] + self.Lij[2][2]
    self.L_ref_csc = self.Lij_csc[0][0] + self.Lij_csc[1][1] + self.Lij_csc[2][2]
    self.L_ref = self.detJabs * self.L_ref[:self.m, :]
    self.L_ref_csc = self.detJabs * self.L_ref_csc[:self.m,:]
    self.T_ref, self.F_ref = self._assemble_TF_full_sigma()
    #self.tau_ref = np.linalg.norm(self.L_ref, ord=2)**2 / np.linalg.norm(self.T_ref, ord=2)**2
    #self.Mgam, self.R, self.Rinv, _ = self._precompute_precond()
    self.tau_ref = None
    self.Mgam=None
    self.R=None
    self.Rinv=None
    
    #self.E_pat = self._compute_sys_pattern(verbose=True)

    # -----------------------------
    # Face-to-face change-of-basis maps (moments) on the reference tet
    # P_mom_ref[face_id][sigmaA][sigmaB] maps lam_B -> lam_A on that face
    # -----------------------------
    self.P_mom_ref, self.M_face_ref = self._precompute_face_P_maps(
      rel_prune=0.0,
      abs_prune=0.0,
      verbose=False
    )
    #self.P_mom_ref_sparse = _dense_to_csc_pruned(self.P_mom_ref)
    #self.M_face_ref_sparse = _dense_to_csc_pruned(self.M_face_ref)

  

  def _precompute_promoted_second_partials(self):
    D = 3
    n = self.n
    q_vol = self.q_vol
    kappa_src = self.kappa_src
    kappa_lap = self.kappa_lap

    D1 = []
    k1 = []
    for i in range(D):
      Di = dmat_build_tprod_natural_pruned(D, n, q_vol, kappa_src, i)
      D1.append(Di)
      k1.append(kappa_src + dk_natural(D, i))

    Lij = [[None for _ in range(D)] for __ in range(D)]
    Lij_csc = [[None for _ in range(D)] for __ in range(D)]
    for i in range(D):
      for j in range(D):
        Dj = dmat_build_tprod_natural_pruned(D, n, q_vol, k1[i], j)
        k2 = k1[i] + dk_natural(D, j)
        D_ij_raw = Dj @ D1[i]
        K = kmat_build_tprod(D, n, q_vol, k2, kappa_lap)
        Lij[i][j] = K @ D_ij_raw
        Lij_csc[i][j] = _dense_to_csc_pruned(Lij[i][j], rel=1e-15, abs_tol=0.0)
    return Lij, Lij_csc 

  def _assemble_TF_full_sigma(self):
    T_blocks = []
    F_blocks = []
    for face_id in range(4):
      sigma = self.face_sigma[face_id]
      fd = self.face[face_id]

      wS_hat = fd["wS_hat"]
      Vt = fd["Vt"]
      n_hat = fd["n_hat"]

      Vv = fd["Vv_sigma"][sigma]
      dVv_hat = fd["dVv_hat_sigma"][sigma]

      n_tilde = self.JinvT @ n_hat
      n_tilde_norm = float(np.linalg.norm(n_tilde))

      wS_phys = wS_hat * (self.detJabs * n_tilde_norm)
      wF_phys = wS_hat * self.detJabs

      Tf = Vt.T @ (wS_phys[:, None] * Vv)

      dVv_x = np.einsum("ab,qmb->qma", self.JinvT, dVv_hat)
      ndot = (n_tilde[0] * dVv_x[:, :, 0] +
              n_tilde[1] * dVv_x[:, :, 1] +
              n_tilde[2] * dVv_x[:, :, 2])
      Ff = Vt.T @ (wF_phys[:, None] * ndot)

      T_blocks.append(Tf)
      F_blocks.append(Ff)

    return np.vstack(T_blocks), np.vstack(F_blocks)

  def _precompute_face_P_maps(self, rel_prune=0.0, abs_prune=0.0, verbose=False):
    """
    Precompute face-to-face moment change-of-basis maps on the reference tet.

    For each face_id and each (sigmaA, sigmaB) in S3 x S3, compute:
      P_mom[face_id][sigmaA][sigmaB] = G(sigmaA,sigmaB) @ inv(M(sigmaB))

    where:
      V_sigma = face basis evaluated on triangle quad points with barycentric permutation sigma
      W       = diag(wS_hat)  (reference surface weights including area_scale)
      M(sigma)= V_sigma^T W V_sigma
      G       = V_sigmaA^T W V_sigmaB

    Interpretation (moments):
      If lamB = ∫ psiB u, then lamA ≈ P_mom * lamB.

    Returns:
      P_mom_ref : dict face_id -> dict sigmaA -> dict sigmaB -> (kf,kf) ndarray
      M_ref     : dict face_id -> dict sigma -> (kf,kf) ndarray   (mass matrices)
    """
    perms = all_S3_perms()

    P_mom_ref = {}
    M_ref = {}

    for face_id in range(4):
      fd = self.face[face_id]

      # Face quad points on reference triangle (u,v)
      Xt = fd["Xt"]           # (nq, 2)
      u = Xt[:, 0]
      v = Xt[:, 1]

      # Reference surface weights (already includes reference face area_scale)
      wS_hat = fd["wS_hat"]   # (nq,)
      Wv = wS_hat[:, None]

      # Face kappa and face-basis structures (same ones used to build fd["Vt"])
      kappa_tri = kappa_face_from_kappa_src(self.kappa_src, face_id)
      alpha_tri, tail_tri, invh_tri = jbasis_build_structures(2, self.n, kappa_tri)

      # Evaluate face basis under each barycentric permutation sigma
      V_sigma = {}
      for sigma in perms:
        u_loc, v_loc = tri_coords_perm(u, v, sigma)
        Xt_sigma = np.stack([u_loc, v_loc], axis=1)
        Vt_sigma = jbasis_eval_all(
          Xt_sigma,
          kappa_tri,
          self.n,
          alpha_tri,
          tail_tri,
          invh_tri,
          2
        )
        V_sigma[sigma] = np.asarray(Vt_sigma, dtype=np.float64)

      # Mass matrices M(sigma) and (optionally) their inverses
      M_face = {}
      Minv_face = {}
      for sigma in perms:
        Vt = V_sigma[sigma]                       # (nq, kf)
        M = Vt.T @ (Wv * Vt)                      # (kf, kf)
        M = 0.5 * (M + M.T)

        # Optional pruning (usually keep off)
        if rel_prune > 0.0 or abs_prune > 0.0:
          fro = float(np.linalg.norm(M))
          thr = max(float(abs_prune), float(rel_prune) * max(1.0, fro))
          if thr > 0.0:
            M = M.copy()
            M[np.abs(M) <= thr] = 0.0

        M_face[sigma] = M
        # Invert with a robust solve (SPD-ish). Use general solve to match your style.
        Minv = scipy.linalg.solve(M, np.eye(self.kf), assume_a="gen", check_finite=False)
        Minv_face[sigma] = Minv

      # Cross Gram and P maps
      P_face = {}
      for sigmaA in perms:
        VA = V_sigma[sigmaA]
        row = {}
        for sigmaB in perms:
          VB = V_sigma[sigmaB]
          G = VA.T @ (Wv * VB)                    # (kf, kf)

          if rel_prune > 0.0 or abs_prune > 0.0:
            fro = float(np.linalg.norm(G))
            thr = max(float(abs_prune), float(rel_prune) * max(1.0, fro))
            if thr > 0.0:
              G = G.copy()
              G[np.abs(G) <= thr] = 0.0

          P = G @ Minv_face[sigmaB]               # (kf, kf)

          if rel_prune > 0.0 or abs_prune > 0.0:
            fro = float(np.linalg.norm(P))
            thr = max(float(abs_prune), float(rel_prune) * max(1.0, fro))
            if thr > 0.0:
              P = P.copy()
              P[np.abs(P) <= thr] = 0.0

          row[sigmaB] = P
        P_face[sigmaA] = row

      P_mom_ref[face_id] = P_face
      M_ref[face_id] = M_face

      if verbose:
        # Quick diagnostics: how close are self-maps to identity?
        # P(sigma <- sigma) should be ~I in exact integration.
        for sigma in perms:
          Pss = P_face[sigma][sigma]
          err = float(np.linalg.norm(Pss - np.eye(self.kf)))
          print(f"[P_mom_ref] face={face_id} sigma={sigma} ||P-I||_F={err:.3e}")

    return P_mom_ref, M_ref



  def _precompute_precond(self):
    alpha_rel=1e-12
    alpha_abs=1e-18
    eps_rel=1e-12
    ## Stacked system
    A = np.vstack([
      np.sqrt(self.tau_ref) * self.T_ref,
      self.L_ref,
    ])
    A = _as_csc(A)
    m, n = A.shape
    deg_of_col = self.alpha_src.sum(axis=1)
    M, _, _, gamma = build_degree_block_pentadiag_M_damped(
      A, deg_of_col,
      alpha_rel=alpha_rel,
      alpha_abs=alpha_abs,
      eps_rel=eps_rel
    )
    R, shift = chol_spd_global(M, eps_rel=eps_rel)
    I = np.eye(n)
    Rinv = scipy.linalg.solve_triangular(R, I, lower=False, check_finite=False)
    return M, R, Rinv, gamma

  def _compute_sys_pattern(self, rel=0.0, abs_tol=0.0, economy=True, verbose=True):
    """
    Build a geometry-agnostic *pattern* matrix Apat for the stacked system
      A = [T_full; L_int]
    by taking unions over:
      - interior blocks Lij (i,j=0..2) with "Gij = 1"
      - trace blocks T_face_sigma over sigma in S3 for each face

    Then run SuiteSparseQR once on Apat to obtain a reusable column ordering E.

    Stores:
      self.Apat_csc : CSC "pattern" matrix with all nonzeros == 1
      self.E_pat    : permutation vector (length nvar)
      self.rank_pat : rank reported by qr(Apat)
      self.R_pat_nnz: nnz(R) from qr(Apat) (useful to estimate fill)
    """
    perms = all_S3_perms()

    # -----------------------------
    # 1) Interior pattern: union_{i,j} pattern(Lij_csc[i][j])
    # -----------------------------
    # Lij_csc entries should already be CSC and pruned numerically.
    Lpat = None
    for i in range(3):
      for j in range(3):
        Aij = self.Lij_csc[i][j]
        # Boolean pattern
        Bij = (Aij != 0)
        Lpat = Bij if Lpat is None else (Lpat + Bij)

    # Keep only interior constraint rows (first m rows), matching your system stacking.
    # Lij_csc are full MxM (promoted full), but your L_int uses [:m, :].
    if Lpat is None:
      raise RuntimeError("Lpat is None; did Lij_csc get built?")
    Lpat = Lpat[:self.m, :]

    # Convert boolean pattern -> numeric 1's
    Lpat = Lpat.astype(np.float64).tocsc()
    if Lpat.nnz:
      Lpat.data[:] = 1.0
      Lpat.eliminate_zeros()

    # -----------------------------
    # 2) Trace pattern: for each face f, union_{sigma} pattern(T_face_sigma_ref_csc[sigma])
    # -----------------------------
    Tpat_blocks = []
    for face_id in range(4):
      d = self.face[face_id]["T_sigma_ref_csc"]

      Tf_pat = None
      for sigma in perms:
        Ts = d[sigma]
        Bs = (Ts != 0)
        Tf_pat = Bs if Tf_pat is None else (Tf_pat + Bs)

      if Tf_pat is None:
        raise RuntimeError(f"Tf_pat is None for face {face_id}")

      Tf_pat = Tf_pat.astype(np.float64).tocsc()
      if Tf_pat.nnz:
        Tf_pat.data[:] = 1.0
        Tf_pat.eliminate_zeros()

      Tpat_blocks.append(Tf_pat)

    Tpat = sps.vstack(Tpat_blocks, format="csc")

    # -----------------------------
    # 3) Full stacked pattern
    # -----------------------------
    Apat = sps.vstack([Tpat, Lpat], format="csc")

    # Optional pruning on the pattern (usually unnecessary; pattern is already exact nonzeros)
    if rel > 0.0 or abs_tol > 0.0:
      fro = float(sps.linalg.norm(Apat))
      thr = max(float(abs_tol), float(rel) * max(1.0, fro))
      if thr > 0.0 and Apat.nnz:
        Apat = Apat.copy()
        mask = np.abs(Apat.data) > thr
        Apat.data = Apat.data[mask]
        Apat.indices = Apat.indices[mask]
        Apat.eliminate_zeros()

    self.Apat_csc = Apat

    if verbose:
      print("[Apat] shape:", Apat.shape,
            "nnz:", Apat.nnz,
            "nnz/col:", Apat.nnz / max(1, Apat.shape[1]))

    # -----------------------------
    # 4) Run SPQR once on Apat to get a reusable ordering
    # -----------------------------
    # Qp, Rp, E, rank satisfy (Matlab convention):
    #   Qp * Rp = Apat[:, E]
    Qp, Rp, E, rank = sparseqr.qr(Apat, economy=economy)

    # Save what we need
    self.E_pat = np.asarray(E, dtype=np.int64)
    self.rank_pat = int(rank)
    self.R_pat_nnz = int(Rp.nnz) if hasattr(Rp, "nnz") else None

    if verbose:
      print("[Apat/spqr] rank:", self.rank_pat,
            "R.nnz:", self.R_pat_nnz)
    return self.E_pat


class TetSteklovLeaf:
  def __init__(self, ref, V_phys, face_sigma,
               rtol_int=1e-14,
               rtol_trace=1e-14):
    self.ref = ref
    self.V_phys = np.asarray(V_phys, dtype=np.float64)
    self.face_out_sign = np.array(
      [tet_face_outward_sign(self.V_phys, f) for f in range(4)],
      dtype=np.float64
    )
    self.face_sigma = list(face_sigma)
    if len(self.face_sigma) != 4:
      raise ValueError("face_sigma must have length 4")
    
    self.face_n_unit = np.zeros((4,3), dtype=np.float64)
    self.face_scale  = np.zeros(4, dtype=np.float64)
    
    for f in range(4):
      tri = LOCAL_FACE_TRIS[f]
      i0, i1, i2 = tri
      opp = [0,1,2,3]
      opp.remove(i0); opp.remove(i1); opp.remove(i2)
      i3 = opp[0]
    
      p0 = self.V_phys[i0]
      p1 = self.V_phys[i1]
      p2 = self.V_phys[i2]
      p3 = self.V_phys[i3]
    
      n = np.cross(p1 - p0, p2 - p0)          # depends only on the face verts
      # outward: normal should point away from opposite vertex
      if float(np.dot(n, p3 - p0)) > 0.0:
        n = -n
    
      nn = float(np.linalg.norm(n))
      self.face_n_unit[f, :] = n / (nn + 1e-300)
      self.face_scale[f] = nn / (self.ref.face_nref_norm[f] + 1e-300)

    


    J, b, detJ, detJabs, Jinv, JinvT, G = tet_affine_from_verts(self.V_phys)
    self.J = J
    self.b = b
    self.detJ = detJ
    self.detJsgn = 1.0 if self.detJ >= 0.0 else -1.0
    self.detJabs = detJabs
    self.JinvT = JinvT
    self.G = G
    self.gverts = None
    # cache for measurement quad (q_err -> (Xhat,w,V))
    self._err_cache = {}

    # affine Laplacian (promoted)
    M = ref.M
    A_lap = np.zeros((M, M), dtype=np.float64)
    for i in range(3):
      for j in range(3):
        gij = float(G[i, j])
        if gij != 0.0:
          A_lap += gij * ref.Lij[i][j]
    self.A_lap = A_lap
    self.L_int = self.detJabs * A_lap[:ref.m, :]
    self.T_full, self.F_full = self._assemble_TF_full_sigma()
    self.A = np.vstack([self.T_full, self.L_int])
    self.Asp = _dense_to_csc_pruned(self.A, rel=1e-14, abs_tol=0.0)
    self.SPQR = self.spqr_factor(self.Asp)
    # -----------------------------
    # Cached sizes / layout helpers
    # -----------------------------
    self.M = int(self.ref.M)        # number of volume dofs
    self.m_int = int(self.ref.m)    # number of interior constraint rows
    self.kf = int(self.ref.kf)      # number of face dofs per face block
    self.nface = 4
    self.nrows = int(self.nface * self.kf + self.m_int)

    # Sanity checks (helpful while wiring merges)
    if self.T_full.shape != (self.nface * self.kf, self.M):
      raise ValueError(f"T_full has shape {self.T_full.shape}, expected {(self.nface*self.kf, self.M)}")
    if self.F_full.shape != (self.nface * self.kf, self.M):
      raise ValueError(f"F_full has shape {self.F_full.shape}, expected {(self.nface*self.kf, self.M)}")
    if self.L_int.shape != (self.m_int, self.M):
      raise ValueError(f"L_int has shape {self.L_int.shape}, expected {(self.m_int, self.M)}")
    if self.Asp.shape != (self.nrows, self.M):
      raise ValueError(f"Asp has shape {self.Asp.shape}, expected {(self.nrows, self.M)}")

    # Cache for identity injection blocks per face (built on demand)
    self._Bface_cache = {}

  def spqr_factor(self, Asp, economy=True):
    """
    Factor Asp (CSC) once:
      Q R = Asp[:, E]    (Matlab convention used by sparseqr)
    Returns a dict you can reuse for many RHS.
    """
    Asp = Asp.tocsc()
    Q, R, E, rank = sparseqr.qr(Asp, economy=economy)
  
    # store in formats good for repeated solves
    Q = Q.tocsc()
    R = R.tocsr()
    E = np.asarray(E, dtype=np.int64)
    rank = int(rank)
    #plt.spy(Q.toarray())
    #plt.show() 
    return {
      "Asp": Asp,
      "Q": Q,
      "R": R,
      "E": E,
      "rank": rank,
      "economy": bool(economy),
    }

  def solve(self, B):
    """
    Solve min ||A X - B||_F with minimum-norm solution.
  
    Parameters
    ----------
    B : array_like
        Either
          - shape (nrows,)          [single RHS]
          - shape (nrows, p)        [multiple RHS]
  
    Returns
    -------
    X : ndarray
        Either
          - shape (nvars,)          if B was 1D
          - shape (nvars, p)        if B was 2D
    """
    F = self.SPQR
    Q = F["Q"]          # (nrows, q)
    R = F["R"]          # (q, nvars), CSR
    E = F["E"]
    r = F["rank"]
  
    B = np.asarray(B, dtype=np.float64)
  
    # --- normalize RHS to 2D ---
    is_vector = (B.ndim == 1)
    if is_vector:
      B = B.reshape(-1, 1)          # (nrows, 1)
  
    # ensure Fortran order for multi-RHS
    if not B.flags.f_contiguous:
      B = np.asfortranarray(B)
  
    nrows, p = B.shape
    nvars = R.shape[1]
  
    # --- apply Q^T ---
    Y = Q.T @ B                     # dense (q, p)
  
    # --- triangular solve on leading block ---
    R11 = R[:r, :r]
    Z = spla.spsolve_triangular(
      R11,
      Y[:r, :],
      lower=False
    )                               # (r, p)
  
    # --- pad in permuted coordinates ---
    Xpre = np.zeros((nvars, p), dtype=np.float64, order="F")
    Xpre[:r, :] = Z
  
    # --- unpermute back to original column order ---
    X = np.empty_like(Xpre)
    X[E, :] = Xpre
  
    # --- return shape consistent with input ---
    if is_vector:
      return X[:, 0]
    return X

  def face_diameter(self, face_id):
    """
    Characteristic size h of a face: max edge length.
    """
    face_id = int(face_id)
    tri = LOCAL_FACE_TRIS[face_id]
    V = self.V_phys
  
    p0 = V[tri[0]]
    p1 = V[tri[1]]
    p2 = V[tri[2]]
  
    h01 = np.linalg.norm(p1 - p0)
    h12 = np.linalg.norm(p2 - p1)
    h20 = np.linalg.norm(p0 - p2)
  
    return max(h01, h12, h20)

  def face_mass_matrix(self, face_id, sparse=False):
    """
    Return the physical face mass matrix M_face (kf x kf) in this leaf's
    face moment basis for the given face_id.
  
    Uses reference precompute:
      ref.M_ref[face_id][sigma] = ∫_face ψ_i ψ_j dS_hat   (includes whatever your ref face measure is)
    and scales by face_scale to get physical dS.
    """
    face_id = int(face_id)
    sigma = self.face_sigma[face_id]
  
    Mref = self.ref.M_face_ref[face_id][sigma]  # dense or sparse depending on your storage
    s = float(self.face_scale[face_id])
  
    if sparse:
      # if Mref is already sparse, keep it sparse
      if sps.issparse(Mref):
        return s * Mref
      return sps.csc_matrix(s * np.asarray(Mref, dtype=np.float64))
    else:
      if sps.issparse(Mref):
        return (s * Mref).toarray()
      return s * np.asarray(Mref, dtype=np.float64)

  def build_face_maps(self, iface, lam_ext_blocks=None, f_int=None):
    """
    Build affine maps on a single face iface:
      mu_iface  = S_F * lam_iface + g_F
      tr_iface  = S_T * lam_iface + g_T
  
    where:
      mu_iface = flux moments on iface
      tr_iface = trace moments on iface (i.e. T_iface c)
  
    Returns:
      S_F: (kf,kf)
      g_F: (kf,)
      S_T: (kf,kf)
      g_T: (kf,)
    """
    iface = int(iface)
    if lam_ext_blocks is None:
      lam_ext_blocks = [None] * self.nface
    if len(lam_ext_blocks) != self.nface:
      raise ValueError("lam_ext_blocks must have length 4")
  
    # g terms: solve with iface Dirichlet = 0 (None treated as zero in your assemble_rhs)
    lam0 = list(lam_ext_blocks)
    lam0[iface] = None
    c0 = self.solve_from_blocks(lam_blocks=lam0, f_int=f_int)
  
    g_F = np.asarray(self.flux_face(iface, c0), dtype=np.float64).reshape(-1)
    g_T = np.asarray(self.trace_face(iface, c0), dtype=np.float64).reshape(-1)
  
    # response maps: solve for all columns of B_face
    B = self.B_face(iface)             # (nrows, kf)
    C = self.solve(B)                  # (Mvol, kf)
  
    Ff = self.F_face(iface)
    Tf = self.T_face(iface)
  
    S_F = (Ff @ C)
    S_T = (Tf @ C)
  
    return S_F, g_F, S_T, g_T

  def build_face_augmented_map(self, iface, tau, lam_ext_blocks=None, f_int=None):
    """
    Build augmented numerical flux map:
      mu_hat = mu + tau * M * (tr - lam)
  
    Returns:
      K_face: (kf,kf)   such that mu_hat = K_face * lam + h_face
      h_face: (kf,)
    """
    iface = int(iface)
    tau = float(tau)
  
    S_F, g_F, S_T, g_T = self.build_face_maps(iface, lam_ext_blocks=lam_ext_blocks, f_int=f_int)
  
    M = self.face_mass_matrix(iface, sparse=False)
    I = np.eye(self.kf)
  
    K_face = S_F + tau * (M @ (S_T - I))
    h_face = g_F + tau * (M @ g_T)
  
    return K_face, h_face   

  def project_neumann_face(self, face_id, grad_u_phys):
    """
    Flux moment vector on a given face:
      mu_l = ∫_face psi_l * (∂u/∂n_out) dS
  
    where n_out is the *physical* outward unit normal for this face.
    grad_u_phys(x,y,z) must return (ux,uy,uz).
    """
    fd = self.ref.face[face_id]
    sigma = self.face_sigma[face_id]
  
    Xf_hat = fd["Xf_hat_sigma"][sigma]
    wS_hat = fd["wS_hat"]
    Vt = fd["Vt"]
    n_hat = fd["n_hat"]
  
    # physical (unnormalized) outward normal is J^{-T} n_hat
    n_tilde = self.JinvT @ n_hat
    n_tilde_norm = float(np.linalg.norm(n_tilde))
  
    # physical surface measure: dS = detJabs * ||J^{-T} n_hat|| dS_hat
    wS_phys = wS_hat * (self.detJabs * n_tilde_norm)
  
    # physical unit normal
    n_out = n_tilde / (n_tilde_norm + 1e-300)
  
    # evaluate grad u on mapped face points
    Xf_phys = map_ref_to_phys(Xf_hat, self.J, self.b)
    ux, uy, uz = grad_u_phys(Xf_phys[:, 0], Xf_phys[:, 1], Xf_phys[:, 2])
  
    ndot = n_out[0] * ux + n_out[1] * uy + n_out[2] * uz
  
    # moment vector in the face basis
    return Vt.T @ (wS_phys * ndot)

  def map_hat_to_phys(self, Xf_hat):
    return map_ref_to_phys(Xf_hat, self.J, self.b)

  def project_u_vol(self, u_phys):
    """
    Weighted L2 projection of u_phys onto the degree-n volume basis on this tet:
      find c s.t.  ∫ phi_i (sum_j c_j phi_j - u) w dV = 0
  
    Discrete form with quadrature:
      M c = b
      M = V^T diag(w) V
      b = V^T diag(w) u
  
    This is robust even when V^T W V != I numerically.
    """
    ref = self.ref
    Xhat = ref.Xhat_vol_src
    w_hat = ref.w_vol_src
    V = ref.V_vol_src  # (nq, M)
  
    # map quad points
    X = map_ref_to_phys(Xhat, self.J, self.b)
    u = u_phys(X[:, 0], X[:, 1], X[:, 2])
  
    # physical volume weights
    w = self.detJabs * w_hat
  
    # assemble mass and rhs
    WV = w[:, None] * V                 # (nq, M)
    Mmat = V.T @ WV                     # (M, M)
    b = V.T @ (w * u)                   # (M,)

    # solve (SPD-ish, but use a general solve)
    c = scipy.linalg.solve(Mmat, b, assume_a="gen", check_finite=False)
    return c, Mmat
  
  def _assemble_TF_full_sigma(self):
    ref = self.ref
    T_blocks = []
    F_blocks = []
  
    for face_id in range(4):
      sigma = self.face_sigma[face_id]
      fd = ref.face[face_id]
  
      # ---------- TRACE (reuse precomputed Tf_ref) ----------
      Tf_ref = fd["T_sigma_ref"][sigma]  # sparse or dense
      sT = float(self.face_scale[face_id])  # dS_phys = face_scale * dS_hat
      T_blocks.append(sT * Tf_ref)
  
      # ---------- FLUX (MUST be built with leaf geometry) ----------
      # Data on the reference face quadrature
      Vt = fd["Vt"]                        # (nq, kf)
      wS_hat = fd["wS_hat"]                # (nq,)
      dVv_hat = fd["dVv_hat_sigma"][sigma] # (nq, Mvol, 3)  gradients in hat coords
  
      # Physical surface weights and outward unit normal (you already computed these)
      wS_phys = wS_hat * self.face_scale[face_id]
      n_unit = self.face_n_unit[face_id]   # (3,)
  
      # Transform gradients to physical coords: ∇_x φ = J^{-T} ∇_hat φ
      dVv_x = np.einsum("ab,qmb->qma", self.JinvT, dVv_hat)  # (nq, Mvol, 3)
  
      # Normal derivative: n · ∇_x φ
      ndot = (n_unit[0] * dVv_x[:, :, 0] +
              n_unit[1] * dVv_x[:, :, 1] +
              n_unit[2] * dVv_x[:, :, 2])  # (nq, Mvol)
  
      # Flux moments: ∫ ψ_i (n·∇φ_j) dS
      Ff = Vt.T @ (wS_phys[:, None] * ndot)  # (kf, Mvol)
      F_blocks.append(Ff)
  
    # Stack
    if sps.issparse(T_blocks[0]):
      T_full = sps.vstack(T_blocks, format="csc")
    else:
      T_full = np.vstack(T_blocks)
  
    # F is dense right now; you can prune/sparsify later if desired
    F_full = np.vstack(F_blocks)
  
    return T_full, F_full
  #def _assemble_TF_full_sigma(self):
  #  ref = self.ref
  #  T_blocks = []
  #  F_blocks = []
  #
  #  for face_id in range(4):
  #    sigma = self.face_sigma[face_id]
  #    fd = ref.face[face_id]
  #
  #    # reference precomputed sparse blocks for this face+sigma
  #    Tf_ref = fd["T_sigma_ref"][sigma]  # (mt_face, Mvol)
  #    Ff_ref = fd["F_sigma_ref"][sigma]  # (mt_face, Mvol)
  #
  #    # geometry scalars
  #    n_hat = fd["n_hat"]
  #    n_tilde = self.JinvT @ n_hat
  #    n_tilde_norm = float(np.linalg.norm(n_tilde))
  #
  #    # matches your original scaling:
  #    # wS_phys = wS_hat * detJabs * ||n_tilde||
  #    # wF_phys = wS_hat * detJabs
  #    #sT = float(self.detJabs * n_tilde_norm)
  #    #sF = float(self.detJabs)
  #    sT = float(self.face_scale[face_id])
  #    sF = float(self.face_scale[face_id]) 
  #    # sparse scaling (preserves sparsity)
  #    T_blocks.append(sT * Tf_ref)
  #    F_blocks.append(sF * Ff_ref)
  #
  #  # sparse vstack
  #  #T_full = sps.vstack(T_blocks, format="csc")
  #  #F_full = sps.vstack(F_blocks, format="csc")
  #  T_full = np.vstack(T_blocks)
  #  F_full = np.vstack(F_blocks)
  #  return T_full, F_full

  def project_source_int(self, f_rhs_phys):
    Xhat = self.ref.Xhat_vol_lap
    w = self.ref.w_vol_lap
    V = self.ref.V_vol_lap
    X = map_ref_to_phys(Xhat, self.J, self.b)
    fv = f_rhs_phys(X[:, 0], X[:, 1], X[:, 2])
    f_lap = V.T @ (w * fv)
    return self.detJabs * f_lap[:self.ref.m]

  def project_dirichlet_face(self, face_id, g_dirichlet_phys):
    fd = self.ref.face[face_id]
    sigma = self.face_sigma[face_id]

    Xf_hat = fd["Xf_hat_sigma"][sigma]
    wS_hat = fd["wS_hat"]
    Vt = fd["Vt"]
    n_hat = fd["n_hat"]

    n_tilde = self.JinvT @ n_hat
    n_tilde_norm = float(np.linalg.norm(n_tilde))
    wS_phys = wS_hat * (self.detJabs * n_tilde_norm)

    Xf_phys = map_ref_to_phys(Xf_hat, self.J, self.b)
    gv = g_dirichlet_phys(Xf_phys[:, 0], Xf_phys[:, 1], Xf_phys[:, 2])
    return Vt.T @ (wS_phys * gv)


  # measurement quad cache
  def _get_err_quad_and_V(self, q_err):
    q_err = int(q_err)
    key = q_err
    if key in self._err_cache:
      return self._err_cache[key]

    Xhat, w = jquad_mapped_build_kappa(3, q_err, self.ref.kappa_src)
    V = jbasis_eval_all(
      Xhat,
      self.ref.kappa_src,
      self.ref.n,
      self.ref.alpha_src,
      self.ref.tail_src,
      self.ref.invh_src,
      3
    )
    self._err_cache[key] = (Xhat, w, V)
    return Xhat, w, V

  def weighted_l2_error_quad(self, c, u_exact_phys, q_err_min):
    """
    Weighted L2 error on this physical tet using a *measurement* quadrature:
      num^2 = detJabs * sum w(q) * (uh(q) - ue(q))^2
      den^2 = detJabs * sum w(q) * (ue(q))^2

    We choose q_err >= q_err_min (caller supplies a safe minimum tied to m_max)
    and also >= assembly q_vol so the error estimate is not underintegrated.
    """
    q_err = max(int(q_err_min), int(self.ref.q_vol))
    Xhat, w, V = self._get_err_quad_and_V(q_err)

    X = map_ref_to_phys(Xhat, self.J, self.b)
    uh = V @ np.asarray(c, dtype=np.float64)
    ue = u_exact_phys(X[:, 0], X[:, 1], X[:, 2])

    diff = uh - ue
    num2 = float(self.detJabs * np.sum(w * diff * diff))
    den2 = float(self.detJabs * np.sum(w * ue * ue))
    return num2, den2

  # -----------------------------
  # RHS layout: slices / offsets
  # -----------------------------
  def face_slice(self, face_id):
    face_id = int(face_id)
    if face_id < 0 or face_id >= self.nface:
      raise ValueError("bad face_id")
    i0 = face_id * self.kf
    return slice(i0, i0 + self.kf)

  def int_slice(self):
    i0 = self.nface * self.kf
    return slice(i0, i0 + self.m_int)

  def rhs_zeros(self, p=None, dtype=np.float64, order="F"):
    """
    Return a zero RHS in stacked RHS layout.
      p=None -> (nrows,)
      p=int  -> (nrows, p)
    """
    if p is None:
      return np.zeros(self.nrows, dtype=dtype)
    return np.zeros((self.nrows, int(p)), dtype=dtype, order=order)

  def _as_2d(self, X):
    """
    Normalize X to a 2D array (n, p). Return (X2d, is_vector).
    """
    X = np.asarray(X, dtype=np.float64)
    is_vector = (X.ndim == 1)
    if is_vector:
      X = X.reshape(-1, 1)
    if not X.flags.f_contiguous:
      X = np.asfortranarray(X)
    return X, is_vector

  def _return_shape_like_input(self, Y2d, is_vector):
    if is_vector:
      return Y2d[:, 0]
    return Y2d

  # -----------------------------
  # Access face operators
  # -----------------------------
  def T_face(self, face_id):
    s = self.face_slice(face_id)
    return self.T_full[s, :]

  def F_face(self, face_id):
    s = self.face_slice(face_id)
    return self.F_full[s, :]

  # -----------------------------
  # Build stacked RHS b = [lam_full; -f_int]
  # Supports vector or many RHS.
  # -----------------------------
  def assemble_rhs(self, lam_blocks=None, f_int=None):
    """
    Assemble stacked RHS b.

    lam_blocks: list of length 4.
      Each entry may be:
        - None  -> treated as zero
        - (kf,) -> vector face moments
        - (kf,p)-> multi-RHS face moments
    f_int:
        - None   -> treated as zero
        - (m,)   -> vector promoted RHS moments
        - (m,p)  -> multi-RHS promoted RHS moments

    Returns:
      b with shape (nrows,) or (nrows,p) depending on inputs.
    """
    if lam_blocks is None:
      lam_blocks = [None] * self.nface
    if len(lam_blocks) != self.nface:
      raise ValueError("lam_blocks must have length 4")

    # Determine p and whether vector-mode
    p = None
    is_vector = True

    # Check lam blocks
    for lb in lam_blocks:
      if lb is None:
        continue
      lb = np.asarray(lb, dtype=np.float64)
      if lb.ndim == 1:
        if lb.size != self.kf:
          raise ValueError("bad lam block size")
      elif lb.ndim == 2:
        if lb.shape[0] != self.kf:
          raise ValueError("bad lam block shape")
        p = lb.shape[1] if p is None else p
        if lb.shape[1] != p:
          raise ValueError("inconsistent p across lam blocks")
        is_vector = False
      else:
        raise ValueError("lam block must be 1D or 2D")

    # Check f_int
    if f_int is not None:
      f_int = np.asarray(f_int, dtype=np.float64)
      if f_int.ndim == 1:
        if f_int.size != self.m_int:
          raise ValueError("bad f_int size")
      elif f_int.ndim == 2:
        if f_int.shape[0] != self.m_int:
          raise ValueError("bad f_int shape")
        p = f_int.shape[1] if p is None else p
        if f_int.shape[1] != p:
          raise ValueError("inconsistent p across rhs")
        is_vector = False
      else:
        raise ValueError("f_int must be 1D or 2D")

    if is_vector:
      b = self.rhs_zeros(p=None)
      for face_id in range(self.nface):
        lb = lam_blocks[face_id]
        if lb is None:
          continue
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        b[self.face_slice(face_id)] = lb
      if f_int is not None:
        b[self.int_slice()] = -np.asarray(f_int, dtype=np.float64).reshape(-1)
      return b

    # multi-RHS case
    if p is None:
      # should not happen if is_vector is False, but keep safe
      raise RuntimeError("multi-RHS detected but p is None")

    b = self.rhs_zeros(p=p)
    for face_id in range(self.nface):
      lb = lam_blocks[face_id]
      if lb is None:
        continue
      lb = np.asarray(lb, dtype=np.float64)
      if lb.ndim == 1:
        b[self.face_slice(face_id), :] = lb.reshape(-1, 1)
      else:
        b[self.face_slice(face_id), :] = lb
    if f_int is not None:
      f_int = np.asarray(f_int, dtype=np.float64)
      if f_int.ndim == 1:
        b[self.int_slice(), :] = -f_int.reshape(-1, 1)
      else:
        b[self.int_slice(), :] = -f_int
    return b

  # -----------------------------
  # Simple injection matrices for interface work
  # -----------------------------
  def B_face(self, face_id):
    """
    Return dense injection matrix B such that:
      b = B @ lam_face
    where lam_face is (kf,) or (kf,p), and b is stacked RHS (nrows,) or (nrows,p)
    with the chosen face block set and everything else zero.
    """
    face_id = int(face_id)
    if face_id in self._Bface_cache:
      return self._Bface_cache[face_id]
    B = np.zeros((self.nrows, self.kf), dtype=np.float64, order="F")
    s = self.face_slice(face_id)
    B[s, :] = np.eye(self.kf, dtype=np.float64)
    self._Bface_cache[face_id] = B
    return B

  # -----------------------------
  # Evaluate trace/flux moments from coefficients
  # -----------------------------
  def trace_face(self, face_id, c):
    """
    Return lambda_face = T_face c
    c: (M,) or (M,p)
    """
    c2d, is_vector = self._as_2d(c)
    out = self.T_face(face_id) @ c2d
    return self._return_shape_like_input(out, is_vector)

  def flux_face(self, face_id, c):
    """
    Return mu_face = F_face c  (outward flux moments)
    c: (M,) or (M,p)
    """
    c2d, is_vector = self._as_2d(c)
    out = self.F_face(face_id) @ c2d
    return self._return_shape_like_input(out, is_vector)

  # -----------------------------
  # Convenience solve wrappers
  # -----------------------------
  def solve_from_blocks(self, lam_blocks=None, f_int=None):
    """
    Assemble RHS from blocks then call solve.
    """
    b = self.assemble_rhs(lam_blocks=lam_blocks, f_int=f_int)
    return self.solve(b)

  def solve_with_face_dirichlet(self, iface, lam_iface, lam_ext_blocks=None, f_int=None):
    """
    Solve local problem with:
      - interface face 'iface' Dirichlet moments set to lam_iface
      - other faces set from lam_ext_blocks (len 4, None allowed; iface entry ignored)
      - interior forcing from f_int

    Returns volume coeffs c.
    """
    iface = int(iface)
    if lam_ext_blocks is None:
      lam_ext_blocks = [None] * self.nface
    if len(lam_ext_blocks) != self.nface:
      raise ValueError("lam_ext_blocks must have length 4")

    lam_blocks = list(lam_ext_blocks)
    lam_blocks[iface] = lam_iface
    return self.solve_from_blocks(lam_blocks=lam_blocks, f_int=f_int)

  # -----------------------------
  # DtN construction on a chosen face
  # -----------------------------
  def build_dtn_face(self, iface, lam_ext_blocks=None, f_int=None):
    """
    Build discrete DtN on a single face iface:
      mu_iface = S * lam_iface + g

    lam_ext_blocks: list length 4, known Dirichlet moments on external faces.
      iface entry is ignored / treated as zero for the g solve.

    f_int: promoted interior RHS moments (m,) (or (m,p) if you want multi-RHS g)

    Returns:
      S: (kf,kf) dense
      g: (kf,)  vector (or (kf,p) if f_int is multi-RHS)
    """
    iface = int(iface)
    if lam_ext_blocks is None:
      lam_ext_blocks = [None] * self.nface
    if len(lam_ext_blocks) != self.nface:
      raise ValueError("lam_ext_blocks must have length 4")

    # 1) g contribution: solve with iface Dirichlet = 0
    lam_blocks0 = list(lam_ext_blocks)
    lam_blocks0[iface] = None
    c0 = self.solve_from_blocks(lam_blocks=lam_blocks0, f_int=f_int)
    g = self.flux_face(iface, c0)

    # 2) S contribution: apply solver to injection matrix for iface
    B = self.B_face(iface)              # (nrows, kf)
    C = self.solve(B)                   # (M, kf)
    S = self.F_face(iface) @ C          # (kf, kf)

    return S, g

  # -----------------------------
  # Diagnostics (optional but recommended)
  # -----------------------------
  def residual_blocks(self, c, lam_blocks=None, f_int=None):
    """
    Return residual blocks:
      r_face = T_full c - lam_full
      r_int  = L_int c + f_int
    """
    c2d, is_vector = self._as_2d(c)

    # lam_full
    if lam_blocks is None:
      lam_blocks = [None] * self.nface
    lam_full = None
    # assemble lam_full as 2D for consistency
    lam_parts = []
    for face_id in range(self.nface):
      lb = lam_blocks[face_id]
      if lb is None:
        lb2d = np.zeros((self.kf, c2d.shape[1]), dtype=np.float64, order="F")
      else:
        lb2d, lb_isvec = self._as_2d(lb)
        if lb2d.shape[0] != self.kf:
          raise ValueError("bad lam block shape")
        if lb2d.shape[1] != c2d.shape[1]:
          # allow broadcasting from vector to p
          if lb2d.shape[1] == 1:
            lb2d = np.repeat(lb2d, c2d.shape[1], axis=1)
          else:
            raise ValueError("lam block p mismatch")
      lam_parts.append(lb2d)
    lam_full = np.vstack(lam_parts)  # (4kf, p)

    r_face = (self.T_full @ c2d) - lam_full

    if f_int is None:
      f2d = np.zeros((self.m_int, c2d.shape[1]), dtype=np.float64, order="F")
    else:
      f2d, f_isvec = self._as_2d(f_int)
      if f2d.shape[0] != self.m_int:
        raise ValueError("bad f_int shape")
      if f2d.shape[1] != c2d.shape[1]:
        if f2d.shape[1] == 1:
          f2d = np.repeat(f2d, c2d.shape[1], axis=1)
        else:
          raise ValueError("f_int p mismatch")

    r_int = (self.L_int @ c2d) + f2d

    return (self._return_shape_like_input(r_face, is_vector),
            self._return_shape_like_input(r_int, is_vector))

  def residual_norms(self, c, lam_blocks=None, f_int=None):
    """
    Return scalar norms (Frobenius if multi-RHS):
      ||T c - lam||, ||L c + f||
    """
    r_face, r_int = self.residual_blocks(c, lam_blocks=lam_blocks, f_int=f_int)

    r_face = np.asarray(r_face, dtype=np.float64)
    r_int = np.asarray(r_int, dtype=np.float64)

    n_face = float(np.linalg.norm(r_face))
    n_int = float(np.linalg.norm(r_int))
    return n_face, n_int

  def interface_flux_jump_norm(self, iface, c_self, other_leaf, other_iface, c_other):
    """
    Compute || mu_self + mu_other ||_2 on the shared interface,
    assuming both flux operators use each tet's outward normal.
    """
    mu0 = np.asarray(self.flux_face(iface, c_self), dtype=np.float64).reshape(-1)
    mu1 = np.asarray(other_leaf.flux_face(other_iface, c_other), dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(mu0 + mu1))

  def interface_trace_mismatch_norm(self, iface, c_self, other_leaf, other_iface, c_other):
    """
    Compute || lambda_self - lambda_other ||_2 on the shared interface
    in the moment basis on each tet. (Assumes both use the same canonical ordering;
    if not, you must align via sigma/permutation at merge level.)
    """
    lam0 = np.asarray(self.trace_face(iface, c_self), dtype=np.float64).reshape(-1)
    lam1 = np.asarray(other_leaf.trace_face(other_iface, c_other), dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(lam0 - lam1))


def make_sym_u_f_and_grad(u_expr, simplify=True):
  """
  Build numpy-callable (u, f=-Δu, grad_u) from a SymPy expression u_expr(x,y,z).

  Inputs:
    u_expr: SymPy expression in symbols x,y,z.
    simplify: if True, simplify f and grad expressions.

  Returns:
    u_fun(x,y,z) -> array
    f_fun(x,y,z) -> array   (f = -Δu)
    grad_u(x,y,z) -> (ux,uy,uz)
  """
  x, y, z = sp.symbols("x y z", real=True)
  u = u_expr

  ux = sp.diff(u, x)
  uy = sp.diff(u, y)
  uz = sp.diff(u, z)

  lap_u = sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(u, z, 2)
  f = -lap_u

  if simplify:
    ux = sp.simplify(ux)
    uy = sp.simplify(uy)
    uz = sp.simplify(uz)
    f  = sp.simplify(f)

  # numpy backend; returns numpy arrays when inputs are numpy arrays
  u_fun = sp.lambdify((x, y, z), u, "numpy")
  f_fun = sp.lambdify((x, y, z), f, "numpy")
  ux_fun = sp.lambdify((x, y, z), ux, "numpy")
  uy_fun = sp.lambdify((x, y, z), uy, "numpy")
  uz_fun = sp.lambdify((x, y, z), uz, "numpy")

  def grad_u_fun(xv, yv, zv):
    return ux_fun(xv, yv, zv), uy_fun(xv, yv, zv), uz_fun(xv, yv, zv)

  return u_fun, f_fun, grad_u_fun

def make_manufactured_u_f_grad(m_max):
  x, y, z = sp.symbols("x y z", real=True)

  # Example 1: a polynomial (degree controlled by m_max)
  # Feel free to replace this with ANY SymPy expression in x,y,z.
  #u_expr = 0
  #for a in range(m_max + 1):
  #  for b in range(m_max + 1 - a):
  #    for c in range(m_max + 1 - a - b):
  #      coef = sp.Rational(1, 1 + a + b + c)
  #      u_expr += coef * (x**a) * (y**b) * (z**c)
  u_expr = sp.exp(sp.cos(x**2 + y**2 + z**2))

  return make_sym_u_f_and_grad(u_expr, simplify=True)  

#def make_manufactured_u_f_grad(m_max):
#  x, y, z = sp.symbols("x y z", real=True)
#
#  # Put a narrow bump near the face x+y+z=1 and near an edge
#  eps = sp.Rational(1, 200)  # try 1/200, 1/500, 1/1000
#  x0, y0, z0 = sp.Rational(9, 10), sp.Rational(5, 100), sp.Rational(5, 100)
#
#  u_expr = sp.exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2) / eps)
#
#  return make_sym_u_f_and_grad(u_expr, simplify=True)

#def make_manufactured_u_f_grad(m_max):
#  x, y, z = sp.symbols("x y z", real=True)
#
#  eps = sp.Rational(1, 15)  # smaller = harder
#  s = 1 - x - y - z
#  u_expr = sp.exp(-s/eps)   # boundary layer at s=0 (the top face)
#
#  return make_sym_u_f_and_grad(u_expr, simplify=True)

#def make_manufactured_u_f_grad(m_max):
#  x, y, z = sp.symbols("x y z", real=True)
#
#  # point near a vertex; choose delta small for nastiness
#  x0, y0, z0 = sp.Rational(1, 100), sp.Rational(1, 100), sp.Rational(1, 100)
#  delta = sp.Rational(1, 1000)  # smaller => more singular-like
#
#  r2 = (x-x0)**2 + (y-y0)**2 + (z-z0)**2 + delta**2
#  u_expr = 1/sp.sqrt(r2)   # smooth due to +delta^2, but very steep
#
#  return make_sym_u_f_and_grad(u_expr, simplify=True)

#def make_manufactured_u_f_grad(m_max):
#  x, y, z = sp.symbols("x y z", real=True)
#  u_expr = sp.Piecewise(
#    (1, x + y + z < sp.Rational(1, 2)),
#    (0, True)
#  ) 
#  return make_sym_u_f_and_grad(u_expr, simplify=True)


def solve_single_tet_min_norm_ls(leaf, u_exact_phys, f_rhs_phys,
                                 w_bc=1.0, w_pde=1.0,
                                 do_print=True):
  """Single-tet solve by *one* minimum-norm least-squares problem.

  We solve for coefficients c (degree-n volume basis) by minimizing
      || w_bc * (T_full c - lam_full) ||_2^2  +  || w_pde * (L_int c + f_int) ||_2^2
  and we take the minimum-norm solution if the stacked system is rank-deficient.

  This matches the OTV/HPS viewpoint: only the *representable* part of boundary
  data can be matched by a discrete harmonic polynomial, but here we also
  include the interior Poisson equations in the same LS solve.
  """
  # Boundary data (all 4 faces) in the same face moment basis as T_full.
  lam_blocks = [leaf.project_dirichlet_face(f, u_exact_phys) for f in range(4)]
  lam_full = np.concatenate(lam_blocks)


  # Interior RHS in the promoted Laplacian basis.
  f_int = leaf.project_source_int(f_rhs_phys)
  print("||lam||", np.linalg.norm(lam_full), "||f_int||", np.linalg.norm(f_int))

  iface = 0
  lam_ext = [leaf.project_dirichlet_face(f, u_exact_phys) for f in range(4)]
  f_int = leaf.project_source_int(f_rhs_phys)
  
  S, g = leaf.build_dtn_face(iface, lam_ext_blocks=lam_ext, f_int=f_int)
  
  # check linearity: mu(lam) = S lam + g
  lam_test = np.random.randn(leaf.kf)
  c = leaf.solve_with_face_dirichlet(iface, lam_test, lam_ext_blocks=lam_ext, f_int=f_int)
  mu = leaf.flux_face(iface, c)
  
  print(np.linalg.norm(mu - (S @ lam_test + g)) / (np.linalg.norm(mu) + 1e-300))




  ## --- Solve with SPQR ---
  b = np.concatenate([
    lam_full,
    -f_int,
  ])
  start_time = time.perf_counter()
  c_spqr = leaf.solve(b)
  end_time = time.perf_counter()
  c = c_spqr
  t_spqr = end_time - start_time
  # c_spqr is returned as a dense numpy array (length n)
  print("[spqr] ||A c - b||2 =", np.linalg.norm(leaf.Asp @ c_spqr - b))
  print("time SPQR =", t_spqr) 

  return c, lam_full


def spy_all_P_mom(ref,
                  rel_prune=1e-14,
                  abs_prune=0.0,
                  pause=0.1):
  """
  Loop over all face_id, sigmaA, sigmaB and spy the sparsity
  pattern of the moment change-of-basis matrices P_mom_ref.

  Parameters
  ----------
  ref : RefTetPrecomp
      Must have ref.P_mom_ref populated.
  rel_prune : float
      Relative pruning threshold passed to _dense_to_csc_pruned.
  abs_prune : float
      Absolute pruning threshold.
  pause : float
      Seconds to pause between plots (set to 0 for manual stepping).
  """
  perms = all_S3_perms()

  for face_id in range(4):
    for sigmaA in perms:
      for sigmaB in perms:
        mat = ref.P_mom_ref[face_id][sigmaA][sigmaB]
        print("P==P^T? : ", np.linalg.norm(mat @ mat.T, ord=2))

        # prune to sparse
        matsp = _dense_to_csc_pruned(
          mat,
          rel=rel_prune,
          abs_tol=abs_prune
        )

        plt.figure()
        plt.spy(matsp.toarray())
        plt.title(
          f"P_mom_ref face={face_id}  "
          f"sigmaA={sigmaA}  sigmaB={sigmaB}\n"
          f"nnz={matsp.nnz}/{matsp.shape[0]**2}"
        )
        plt.xlabel("j (from sigmaB)")
        plt.ylabel("i (to sigmaA)")
        plt.tight_layout()
        plt.close()
        #plt.show(block=False)

        #if pause > 0:
          #plt.pause(1)
          #plt.close()
        #else:
        #  input("Press Enter for next plot...")



def run_single_tet_poly_convergence(m_max=8, n_max=14, q_pad=2,
                                   kappa_src=None,
                                   rtol_int=1e-14,
                                   rtol_trace=1e-14,
                                   do_print=True,
                                   method="minls",
                                   w_bc=1.0,
                                   w_pde=1.0):
  """Convergence test on a single physical tet ."""
  if kappa_src is None:
    kappa_src = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
  else:
    kappa_src = np.asarray(kappa_src, dtype=np.float64)

  u_exact, f_rhs, grad_u = make_manufactured_u_f_grad(m_max) #make_poly_u_f_and_grad(m_max)

  # One physical tet (same as tetA in the two-tet test)
  v0 = np.array([0.20, -0.10, 0.30])
  v1 = np.array([1.10,  0.05, 0.20])
  v2 = np.array([0.10,  1.00, 0.40])
  v3 = np.array([0.25,  0.20, 1.40])
  #v0 = np.array([0.0,0.0,0.0])
  #v1 = np.array([1.0,0.0,0.0])
  #v2 = np.array([0.0,1.0,0.0])
  #v3 = np.array([0.0,0.0,1.0])
  VA = np.stack([v0, v1, v2, v3], axis=0)

  tetA_g = (2, 1, 3, 0)
  #tetA_g = (10, 3, 5, 57)
  sigA = build_face_sigma_list_for_tet(tetA_g)

  q_err_min = int(m_max + q_pad)

  dofs = []
  errs = []

  for n in range(10, n_max + 1):
    q_vol = n + q_pad
    q_face = n + q_pad
    ref = RefTetPrecomp(n=n, q_vol=q_vol, q_face=q_face, kappa_src=kappa_src)
    #spy_all_P_mom(
    #  ref,
    #  rel_prune=1e-14,
    #  abs_prune=0.0,
    #  pause=0.05
    #)
    leaf = TetSteklovLeaf(ref, VA, face_sigma=sigA, rtol_int=rtol_int, rtol_trace=rtol_trace)

    if method == "minls":
      c, lam_full = solve_single_tet_min_norm_ls(
        leaf, u_exact, f_rhs,
        w_bc=w_bc, w_pde=w_pde,
        do_print=(do_print)
      )
      print("||c||", np.linalg.norm(c))
      # tail energy: last 20% modes (or last block)
      M = c.size
      i0 = int(0.8 * M)
      print("tail_frac", np.linalg.norm(c[i0:]) / (np.linalg.norm(c) + 1e-300))
    else:
      raise ValueError("unknown method: " + str(method))
    rel_err = leaf.weighted_l2_error_quad(c, u_exact, q_err_min=q_err_min)
    if isinstance(rel_err, tuple) or isinstance(rel_err, list):
      rel_err = rel_err[0]
    dofs.append(int(ref.m))
    errs.append(float(rel_err))

    if do_print:
      print(f"n={n:2d}  dofs={ref.m:5d}  rel_err={rel_err:.3e}")

  return np.array(dofs, dtype=np.int64), np.array(errs, dtype=np.float64)


if __name__ == "__main__":
  dofs, errs = run_single_tet_poly_convergence(m_max=12, n_max=17, q_pad=1, do_print=True, method="minls")
  s = np.zeros_like(errs)
  for j in range(len(errs)):
    if j == 0:
      s[j] = np.log(errs[j])
    else:
      s[j] = np.log(errs[j])-np.log(errs[j-1])
  plt.semilogy(dofs, errs)
  plt.show()
