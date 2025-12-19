# test_merge_two_hps_trace_from_TN_l2err_hiQ.py
#
# Fully updated merge_two Steklov/HPS code with:
#
#   (A) L2 error computed by QUADRATURE EVALUATION (not Parseval/projection):
#         uh = V_src(q_err) @ c
#         ue = u_exact_phys evaluated at mapped physical quad points
#       then ||uh-ue|| / ||ue|| in the (kappa_src) weighted L2 norm.
#
#       IMPORTANT: we use a “measurement quadrature” that is at least high
#       enough for the manufactured solution degree m_max (and also not lower
#       than the assembly quadrature). This avoids the error estimate being
#       dominated by an underintegrated norm.
#
#   (B) Correct trace-space diagnostic on the RIGHT OBJECT:
#         lam_harm := lam_full - T_full @ cp
#       and projection residual:
#         ||lam_harm - U(U^T lam_harm)|| / ||lam_harm||
#
#   (C) Boundary enforcement residual:
#         ||lam_full - T_full c|| / ||lam_full||
#
# Two-space indentation everywhere.

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from jdmat import dmat_build_tprod_natural_pruned
from jkmat import kmat_build_tprod
from jquad_tprod import jquad_mapped_build_kappa
from jbasis import (
  jbasis_build_structures,
  jbasis_eval_all,
  jbasis_eval_all_with_grad,
)


def dim_Pi(D, n):
  from math import comb
  return comb(n + D, D)


def selectors_for_iface(kf, iface):
  """Return (S_I, S_E, ext_faces).
  S_I: (kf x 4kf) selects the interface face block.
  S_E: (3kf x 4kf) selects the three exterior face blocks stacked in increasing face order.
  """
  kfull = 4 * kf
  SI = np.zeros((kf, kfull), dtype=np.float64)
  SE = np.zeros((3 * kf, kfull), dtype=np.float64)
  SI[:, iface * kf:(iface + 1) * kf] = np.eye(kf, dtype=np.float64)
  ext = [f for f in range(4) if f != iface]
  for j, f in enumerate(ext):
    SE[j * kf:(j + 1) * kf, f * kf:(f + 1) * kf] = np.eye(kf, dtype=np.float64)
  return SI, SE, ext

def dk_natural(D, axis):
  dk = np.zeros(D + 1, dtype=np.float64)
  dk[axis] = 1.0
  dk[D] = 1.0
  return dk


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


def interior_qr_solve_cp(fact, rhs):
  Q = fact["Q"]
  R11 = fact["R11"]
  piv = fact["piv"]
  m = fact["m"]
  M = fact["M"]

  rhs = np.asarray(rhs, dtype=np.float64)
  if rhs.shape != (m,):
    raise ValueError("rhs must have shape (m,)")

  y = Q.T @ rhs
  x = scipy.linalg.solve_triangular(R11, y[:m], lower=False, check_finite=False)

  cp = np.zeros(M, dtype=np.float64)
  cp[piv[:m]] = x
  return cp


LOCAL_FACE_TRIS = [
  (1, 2, 3),
  (0, 2, 3),
  (0, 1, 3),
  (0, 1, 2),
]


def all_S3_perms():
  return [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
  ]


def tri_coords_perm(u, v, sigma):
  u = np.asarray(u, dtype=np.float64)
  v = np.asarray(v, dtype=np.float64)
  l0 = 1.0 - u - v
  l1 = u
  l2 = v
  l = [l0, l1, l2]
  return l[sigma[1]], l[sigma[2]]


def tet_local_face_from_global_triple(tet_gverts, face_gtriple_set):
  tg = list(tet_gverts)
  s = set(face_gtriple_set)
  for face_id, tri in enumerate(LOCAL_FACE_TRIS):
    gtri = {tg[tri[0]], tg[tri[1]], tg[tri[2]]}
    if gtri == s:
      return face_id
  raise ValueError("face not found in tet")


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


class RefTetPrecomp:
  def __init__(self, n, q_vol, q_face, kappa_src):
    self.D = 3
    self.n = int(n)
    self.q_vol = int(q_vol)
    self.q_face = int(q_face)
    self.kappa_src = np.asarray(kappa_src, dtype=np.float64)

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
      for sigma in perms:
        u_loc, v_loc = tri_coords_perm(u, v, sigma)
        Xf_hat, _, _ = face_map_and_geom(face_id, u_loc, v_loc)
        Vv, dVv_hat = jbasis_eval_all_with_grad(
          Xf_hat, self.kappa_src, self.n, alpha_src, tail_src, invh_src, 3
        )
        Xf_hat_sigma[sigma] = Xf_hat
        Vv_sigma[sigma] = Vv
        dVv_hat_sigma[sigma] = dVv_hat

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
      })

    self.Lij = self._precompute_promoted_second_partials()

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
    for i in range(D):
      for j in range(D):
        Dj = dmat_build_tprod_natural_pruned(D, n, q_vol, k1[i], j)
        k2 = k1[i] + dk_natural(D, j)
        D_ij_raw = Dj @ D1[i]
        K = kmat_build_tprod(D, n, q_vol, k2, kappa_lap)
        Lij[i][j] = K @ D_ij_raw
    return Lij


class TetSteklovLeaf:
  def __init__(self, ref, V_phys, face_sigma,
               rtol_int=1e-14,
               rtol_trace=1e-14):
    self.ref = ref
    self.V_phys = np.asarray(V_phys, dtype=np.float64)
    self.face_sigma = list(face_sigma)
    if len(self.face_sigma) != 4:
      raise ValueError("face_sigma must have length 4")

    J, b, detJ, detJabs, Jinv, JinvT, G = tet_affine_from_verts(self.V_phys)
    self.J = J
    self.b = b
    self.detJ = detJ
    self.detJabs = detJabs
    self.JinvT = JinvT
    self.G = G

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
    self.L_int = A_lap[:ref.m, :]

    self.int_fact = interior_qr_factor(self.L_int, rtol=rtol_int)
    self.N = self.int_fact["N"]  # (M,k)

    self.T_full, self.F_full = self._assemble_TF_full_sigma()

    # admissible trace basis from A_full = T_full N
    A_full = self.T_full @ self.N  # (4*kf,k)
    s = np.linalg.svd(A_full, compute_uv=False)
    if s.size == 0:
      raise RuntimeError("empty singular spectrum for A_full")
    smax = float(s[0])
    smin = float(s[-1])
    if smax == 0.0:
      raise RuntimeError("A_full is all-zero (unexpected)")
    if smin <= rtol_trace * smax:
      raise RuntimeError(f"A_full rank-deficient: smin/smax={smin/smax:.3e} (face quad too low?)")

    self.U, self.RA = np.linalg.qr(A_full, mode="reduced")  # U: (4*kf,k)

    self.T = self.U.T @ self.T_full
    self.F = self.U.T @ self.F_full

    self.A = self.T @ self.N
    self.B = self.F @ self.N

    self.A_lu, self.A_piv = scipy.linalg.lu_factor(self.A, check_finite=False)

    k = ref.k
    AinvI = scipy.linalg.lu_solve(
      (self.A_lu, self.A_piv),
      np.eye(k, dtype=np.float64),
      check_finite=False
    )
    self.Sred = self.B @ AinvI

  def face_moments_flux(self, face_id, grad_u_phys):
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



  def gauge_particular_zero_trace(self, cp0):
    # Enforce T cp = 0 by shifting cp within cp + Range(N)
    p0 = self.T @ cp0
    y0 = scipy.linalg.lu_solve((self.A_lu, self.A_piv), -p0, check_finite=False)
    return cp0 + (self.N @ y0)


  def gauge_cp_full_trace_min(self, cp0):
    """Gauge the particular solution within cp0 + Range(N) by minimizing full
    face-trace moments in the *moment coordinates* produced by T_full.

      y_g = argmin_y || (T_full N) y + T_full cp0 ||_2
      cp  = cp0 + N y_g

    This makes exterior Dirichlet completion AE y = lam_hE consistent, because it
    removes the component of T_full cp0 that lies outside Range(T_full N).
    """
    cp0 = np.asarray(cp0, dtype=np.float64)
    A_full = self.T_full @ self.N          # (4*kf, k)
    rhs = -(self.T_full @ cp0)             # (4*kf,)
    y_g, *_ = np.linalg.lstsq(A_full, rhs, rcond=None)
    return cp0 + (self.N @ y_g)


  def _assemble_TF_full_sigma(self):
    ref = self.ref
    T_blocks = []
    F_blocks = []
    for face_id in range(4):
      sigma = self.face_sigma[face_id]
      fd = ref.face[face_id]

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

  def project_f_int(self, f_rhs_phys):
    Xhat = self.ref.Xhat_vol_lap
    w = self.ref.w_vol_lap
    V = self.ref.V_vol_lap
    X = map_ref_to_phys(Xhat, self.J, self.b)
    fv = f_rhs_phys(X[:, 0], X[:, 1], X[:, 2])
    f_lap = V.T @ (w * fv)
    return f_lap[:self.ref.m]

  def compute_cp_from_f(self, f_int):
    return interior_qr_solve_cp(self.int_fact, -np.asarray(f_int, dtype=np.float64))

  def compute_b_red_from_cp(self, cp):
    p = self.T @ cp
    q = self.F @ cp
    Ainv_p = scipy.linalg.lu_solve((self.A_lu, self.A_piv), p, check_finite=False)
    return q - (self.B @ Ainv_p)

  def full_lambda_from_face_blocks(self, face_blocks):
    kf = self.ref.kf
    lam = np.zeros(4 * kf, dtype=np.float64)
    for face_id in range(4):
      i0 = face_id * kf
      i1 = i0 + kf
      lam[i0:i1] = face_blocks[face_id]
    return lam

  def solve_c_from_lambda_full(self, lambda_full, cp):
    lambda_full = np.asarray(lambda_full, dtype=np.float64)
    hatlambda = self.U.T @ lambda_full
    rhs = hatlambda - (self.T @ cp)
    y = scipy.linalg.lu_solve((self.A_lu, self.A_piv), rhs, check_finite=False)
    return cp + (self.N @ y)

  def face_moments_dirichlet(self, face_id, g_dirichlet_phys):
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

  def iface_flux_moments_exact(self, face_id, grad_u_exact_phys):
    fd = self.ref.face[face_id]
    sigma = self.face_sigma[face_id]

    Xf_hat = fd["Xf_hat_sigma"][sigma]
    wS_hat = fd["wS_hat"]
    Vt = fd["Vt"]
    n_hat = fd["n_hat"]

    n_tilde = self.JinvT @ n_hat
    n_tilde_norm = float(np.linalg.norm(n_tilde))
    wS_phys = wS_hat * (self.detJabs * n_tilde_norm)

    n_out = n_tilde / (n_tilde_norm + 1e-300)

    Xf_phys = map_ref_to_phys(Xf_hat, self.J, self.b)
    ux, uy, uz = grad_u_exact_phys(Xf_phys[:, 0], Xf_phys[:, 1], Xf_phys[:, 2])
    ndot = n_out[0] * ux + n_out[1] * uy + n_out[2] * uz
    return Vt.T @ (wS_phys * ndot)

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


#def make_poly_u_f_and_grad(m_max):
#
#  def u_fun(x, y, z):
#    return np.exp(x**2 + y**2 + z**2)
#  
#  def f_fun(x, y, z):
#    return 2.0 * np.exp(x**2+y**2+z**2) * (3+2*x**2+2*y**2+2*z**2)
#
#  def grad_u(x, y, z):
#    term = 2*np.exp(x**2+y**2+z**2)
#    ux = x * term
#    uy = y * term
#    uz = z * term
#    return ux, uy, uz
#  
#  return u_fun, f_fun, grad_u

#
#def make_poly_u_f_and_grad(m_max):
#  terms = []
#  for a in range(m_max + 1):
#    for b in range(m_max + 1 - a):
#      for c in range(m_max + 1 - a - b):
#        coef = 1.0 / (1.0 + a + b + c)
#        terms.append((a, b, c, coef))
#
#  def u_fun(x, y, z):
#    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
#    out = np.zeros_like(x, dtype=np.float64)
#    for a, b, c, coef in terms:
#      out += coef * (x ** a) * (y ** b) * (z ** c)
#    return out
#
#  def f_fun(x, y, z):
#    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
#    out = np.zeros_like(x, dtype=np.float64)
#    for a, b, c, coef in terms:
#      if a >= 2:
#        out -= coef * (a * (a - 1)) * (x ** (a - 2)) * (y ** b) * (z ** c)
#      if b >= 2:
#        out -= coef * (b * (b - 1)) * (x ** a) * (y ** (b - 2)) * (z ** c)
#      if c >= 2:
#        out -= coef * (c * (c - 1)) * (x ** a) * (y ** b) * (z ** (c - 2))
#    return out
#
#  def grad_u(x, y, z):
#    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
#    ux = np.zeros_like(x, dtype=np.float64)
#    uy = np.zeros_like(x, dtype=np.float64)
#    uz = np.zeros_like(x, dtype=np.float64)
#    for a, b, c, coef in terms:
#      if a >= 1:
#        ux += coef * a * (x ** (a - 1)) * (y ** b) * (z ** c)
#      if b >= 1:
#        uy += coef * b * (x ** a) * (y ** (b - 1)) * (z ** c)
#      if c >= 1:
#        uz += coef * c * (x ** a) * (y ** b) * (z ** (c - 1))
#    return ux, uy, uz
#
#  return u_fun, f_fun, grad_u

import sympy as sp

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

#def make_manufactured_u_f_grad(m_max):
#  x, y, z = sp.symbols("x y z", real=True)
#
#  # Example 1: a polynomial (degree controlled by m_max)
#  # Feel free to replace this with ANY SymPy expression in x,y,z.
#  #u_expr = 0
#  #for a in range(m_max + 1):
#  #  for b in range(m_max + 1 - a):
#  #    for c in range(m_max + 1 - a - b):
#  #      coef = sp.Rational(1, 1 + a + b + c)
#  #      u_expr += coef * (x**a) * (y**b) * (z**c)
#  u_expr = sp.exp(x**2 + y**2 + z**2)
#
#  return make_sym_u_f_and_grad(u_expr, simplify=True)  

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

def make_manufactured_u_f_grad(m_max):
  x, y, z = sp.symbols("x y z", real=True)

  eps = sp.Rational(1, 10)  # smaller = harder
  s = 1 - x - y - z
  u_expr = sp.exp(-s/eps)   # boundary layer at s=0 (the top face)

  return make_sym_u_f_and_grad(u_expr, simplify=True)

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


def nullspace_svd_rows(C, rtol=None):
  """Return an orthonormal basis B for ker(C) where C is (p x M).

  Uses SVD with a machine-precision scaled tolerance by default:
    tol = eps * max(p, M) * smax
  This is robust for ill-conditioned trace maps at high n and preserves the
  expected bubble dimension dim ker(T_full)=C(n-1,3) for tets (n>=4).
  """
  C = np.asarray(C, dtype=np.float64)
  p, M = C.shape
  if p == 0:
    return np.eye(M, dtype=np.float64)
  # Full SVD of C
  U, s, Vt = np.linalg.svd(C, full_matrices=False)
  if s.size == 0:
    return np.eye(M, dtype=np.float64)
  smax = float(s[0])
  if rtol is None:
    tol = np.finfo(np.float64).eps * max(p, M) * smax
  else:
    tol = float(rtol) * smax
  r = int(np.sum(s > tol))
  B = Vt[r:, :].T  # (M, M-r)
  # Columns of V are orthonormal already
  if B.size == 0:
    return np.zeros((M, 0), dtype=np.float64)
  return B




def solve_single_tet_bubble_harmonic(leaf, u_exact_phys, f_rhs_phys,
                                    rtol_bubble=1e-14,
                                    do_print=True):
  """Single-tet solve: cp in bubble space (T_full cp = 0) + harmonic completion to match full Dirichlet trace."""
  # Interior RHS in promoted basis
  f_int = leaf.project_f_int(f_rhs_phys)

  # Bubble basis for ALL faces: ker(T_full)
  B = nullspace_svd_rows(leaf.T_full, rtol=None)  # (M, mb)

  if do_print:
    print(f"[bubble] B.shape={B.shape}  M={leaf.T_full.shape[1]}  rows(T)={leaf.T_full.shape[0]}")
  # Particular cp: solve L_int (B x) = -f_int in LS sense
  if B.shape[1] == 0:
    # No bubble space (very low n): fall back to existing particular solver (will not satisfy T cp = 0).
    cp = leaf.compute_cp_from_f(f_int)
  else:
    Lb = leaf.L_int @ B
    rhs = -np.asarray(f_int, dtype=np.float64)
    x, *_ = np.linalg.lstsq(Lb, rhs, rcond=None)
    cp = B @ x

  # Check homogeneous Dirichlet for cp
  t_cp = leaf.T_full @ cp
  rel_t_cp = float(np.linalg.norm(t_cp) / (np.linalg.norm(t_cp) + 1.0)) if t_cp.size else 0.0
  # Better: normalize by boundary data norm
  lam_blocks = [leaf.face_moments_dirichlet(f, u_exact_phys) for f in range(4)]
  lam_full = np.concatenate(lam_blocks)
  rel_t_cp2 = float(np.linalg.norm(t_cp) / (np.linalg.norm(lam_full) + 1e-300))

  if do_print:
    print(f"[cp] ||T cp||/||lam||={rel_t_cp2:.3e}  ||T cp||={np.linalg.norm(t_cp):.3e}  ||lam||={np.linalg.norm(lam_full):.3e}")


  # Harmonic completion (OTV/HPS local trace solve):
  # We must only enforce the representable trace subspace Range(T_full N).
  A = leaf.T_full @ leaf.N  # (4*kf x k)

  # SVD-based range compression and stable pseudoinverse solve.
  U, s, Vt = np.linalg.svd(A, full_matrices=False)
  if s.size == 0:
    y = np.zeros((leaf.N.shape[1],), dtype=np.float64)
    rel_res = 0.0
    rel_unrep = 0.0
  else:
    smax = float(s[0])
    tol = np.finfo(np.float64).eps * max(A.shape) * smax
    r = int(np.sum(s > tol))

    # Representable component of lam_full in Range(A)
    lam_hat = U[:, :r].T @ lam_full
    # Least-norm y that matches the representable component exactly:
    y = (Vt[:r, :].T) @ (lam_hat / s[:r])

    # Diagnostics: representable residual and unrepresentable fraction
    res = (A @ y) - lam_full
    rel_res = float(np.linalg.norm(res) / (np.linalg.norm(lam_full) + 1e-300))

    lam_rep = U[:, :r] @ (U[:, :r].T @ lam_full)
    lam_unrep = lam_full - lam_rep
    rel_unrep = float(np.linalg.norm(lam_unrep) / (np.linalg.norm(lam_full) + 1e-300))

  c_h = leaf.N @ y
  rel_harm = float(np.linalg.norm(leaf.L_int @ c_h) / (np.linalg.norm(c_h) + 1e-300))
  c = cp + c_h
  rel_bc = float(np.linalg.norm(leaf.T_full @ c - lam_full) / (np.linalg.norm(lam_full) + 1e-300))

  if do_print:
    print(f"[harm] rel_bc={rel_bc:.3e} rel_res={rel_res:.3e} rel_unrep={rel_unrep:.3e} rel_harm={rel_harm:.3e}")

  return c, cp, B, lam_full


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
  lam_blocks = [leaf.face_moments_dirichlet(f, u_exact_phys) for f in range(4)]
  lam_full = np.concatenate(lam_blocks)


  # Interior RHS in the promoted Laplacian basis.
  f_int = leaf.project_f_int(f_rhs_phys)
  print("||lam||", np.linalg.norm(lam_full), "||f_int||", np.linalg.norm(f_int))

  A_full = leaf.T_full @ leaf.N
  U, s, _ = np.linalg.svd(A_full, full_matrices=False)
  
  # choose numerical rank (don’t use eps here; use something tied to your goals)
  rtol = 1e-10
  r = np.sum(s > rtol * s[0])
  Ur = U[:, :r]
  
  Tproj = Ur.T @ leaf.T_full
  lam_proj = Ur.T @ lam_full

  #A = np.vstack([
  #  w_bc * Tproj,
  #  w_pde * leaf.L_int,
  #])
  #b = np.concatenate([
  #  w_bc * lam_proj,
  #  -w_pde * f_int,
  #])

  plt.spy(np.abs(leaf.T_full)/np.linalg.norm(leaf.T_full,'fro')>1e-14)
  #plt.spy(np.abs(leaf.T_full) > 1e-14)
  plt.show()

  ## Stacked system
  T = leaf.T_full
  L = leaf.L_int
  A = np.vstack([
    w_bc * T,
    w_pde * L,
  ])
  b = np.concatenate([
    w_bc * lam_full,
    -w_pde * np.asarray(f_int, dtype=np.float64),
  ])

  # Minimum-norm LS (SVD-based driver).
  # scipy.linalg.lstsq(..., lapack_driver='gelsd') returns the min-norm solution
  # in rank-deficient cases.
  c, resid, rnk, s = scipy.linalg.lstsq(A, b, lapack_driver="gelsd", check_finite=False)

  # Diagnostics
  bc_res = Ur.T @ (T @ c) - Ur.T @ lam_full#T @ c - lam_full
  pde_res = L @ c + np.asarray(f_int, dtype=np.float64)

  rel_bc = float(np.linalg.norm(bc_res) / (np.linalg.norm(Ur.T @ lam_full) + 1e-300))
  rel_pde = float(np.linalg.norm(pde_res) / (np.linalg.norm(f_int) + 1e-300))
  # How much of lam_full is outside Range(T N)
  A_harm = T @ leaf.N
  Uh, sh, _ = np.linalg.svd(A_harm, full_matrices=False)
  if sh.size == 0 or sh[0] == 0.0:
    rel_unrep = 0.0
  else:
    tol = np.finfo(np.float64).eps * max(A_harm.shape) * float(sh[0])
    rr = int(np.sum(sh > tol))
    lam_rep = Uh[:, :rr] @ (Uh[:, :rr].T @ lam_full)
    rel_unrep = float(np.linalg.norm(lam_full - lam_rep) / (np.linalg.norm(lam_full) + 1e-300))

  if do_print:
    smin = float(s[-1]) if isinstance(s, np.ndarray) and s.size else 0.0
    smax = float(s[0]) if isinstance(s, np.ndarray) and s.size else 0.0
    cond = (smax / (smin + 1e-300)) if smax > 0 else 0.0
    print(f"[minls] rank={int(rnk)}/{min(A.shape)} cond~{cond:.3e}  "
          f"rel_bc={rel_bc:.3e} rel_pde={rel_pde:.3e} rel_unrep={rel_unrep:.3e}")

  return c, lam_full





def run_single_tet_poly_convergence(m_max=8, n_max=14, q_pad=2,
                                   kappa_src=None,
                                   rtol_int=1e-14,
                                   rtol_trace=1e-14,
                                   rtol_bubble=1e-14,
                                   do_print=True,
                                   method="minls",
                                   w_bc=1.0,
                                   w_pde=1.0):
  """Convergence test on a single physical tet using bubble cp + harmonic completion."""
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
  VA = np.stack([v0, v1, v2, v3], axis=0)

  tetA_g = (0, 1, 2, 3)
  sigA = build_face_sigma_list_for_tet(tetA_g)

  q_err_min = int(m_max + q_pad)

  dofs = []
  errs = []

  for n in range(2, n_max + 1):
    q_vol = n + q_pad
    q_face = n + q_pad
    ref = RefTetPrecomp(n=n, q_vol=q_vol, q_face=q_face, kappa_src=kappa_src)
    leaf = TetSteklovLeaf(ref, VA, face_sigma=sigA, rtol_int=rtol_int, rtol_trace=rtol_trace)

    if method == "bubble":
      c, cp, B, lam_full = solve_single_tet_bubble_harmonic(
        leaf, u_exact, f_rhs,
        rtol_bubble=rtol_bubble,
        do_print=(do_print)
      )
    elif method == "minls":
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
      cp = None
      B = np.zeros((ref.M, 0), dtype=np.float64)
    else:
      raise ValueError("unknown method: " + str(method))
    rel_err = leaf.weighted_l2_error_quad(c, u_exact, q_err_min=q_err_min)
    if isinstance(rel_err, tuple) or isinstance(rel_err, list):
      rel_err = rel_err[0]
    dofs.append(int(ref.m))
    errs.append(float(rel_err))

    if do_print:
      print(f"n={n:2d}  dofs={ref.m:5d}  mb={B.shape[1]:5d}  rel_err={rel_err:.3e}")

  return np.array(dofs, dtype=np.int64), np.array(errs, dtype=np.float64)


if __name__ == "__main__":
  # Default: manufactured degree 8, should snap near roundoff once n >= 8 with sufficient quadrature.
  dofs, errs = run_single_tet_poly_convergence(m_max=12, n_max=14, q_pad=1, do_print=True, method="minls")
  s = np.zeros_like(errs)
  for j in range(len(errs)):
    if j == 0:
      s[j] = np.log(errs[j])
    else:
      s[j] = np.log(errs[j])-np.log(errs[j-1])
  print(s)
  plt.semilogy(dofs, errs)
  plt.show()
