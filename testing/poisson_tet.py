import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg
import sympy as sp
from math import comb

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

def _as_csc(A):
  if sps.issparse(A):
    return A.tocsc()
  return np.asarray(A)

def _degree_blocks(deg_of_col):
  deg_of_col = np.asarray(deg_of_col, dtype=np.int64)
  degs = np.unique(deg_of_col)
  blocks = []
  for d in degs:
    J = np.where(deg_of_col == d)[0]
    if J.size:
      blocks.append((int(d), J))
  # assume graded ordering; still fine if not
  blocks.sort(key=lambda t: t[0])
  return blocks


def _block_diag_from_A(A, Jd):
  # Returns H = A[:,Jd]^T A[:,Jd] as dense
  if sps.issparse(A):
    Ad = A[:, Jd]
    return (Ad.T @ Ad).toarray()
  Ad = A[:, Jd]
  return Ad.T @ Ad


def _block_offdiag_from_A(A, Jd, Jdp1):
  # Returns K = A[:,Jd]^T A[:,Jdp1] as dense
  if sps.issparse(A):
    Ad = A[:, Jd]
    B = A[:, Jdp1]
    return (Ad.T @ B).toarray()
  Ad = A[:, Jd]
  B = A[:, Jdp1]
  return Ad.T @ B

def _AtA_block(A, Jrow, Jcol):
  # Return dense block (A[:,Jrow]^T A[:,Jcol])
  if sps.issparse(A):
    Ar = A[:, Jrow]
    Ac = A[:, Jcol]
    return (Ar.T @ Ac).toarray()
  Ar = A[:, Jrow]
  Ac = A[:, Jcol]
  return Ar.T @ Ac

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

def degree_block_norms_of_normal(A, deg_of_col, max_delta=4):
  
  A = A.tocsc() if sps.issparse(A) else np.asarray(A)
  AtA = (A.T @ A).toarray() if sps.issparse(A) else (A.T @ A)

  deg = np.asarray(deg_of_col, dtype=np.int64)
  dmax = int(deg.max())

  J = [np.where(deg == d)[0] for d in range(dmax + 1)]

  out = {}
  for delta in range(max_delta + 1):
    s = 0.0
    cnt = 0
    for d in range(dmax + 1 - delta):
      I = J[d]
      K = J[d + delta]
      if I.size == 0 or K.size == 0:
        continue
      blk = AtA[np.ix_(I, K)]
      s += np.linalg.norm(blk, ord="fro")**2
      cnt += 1
    out[delta] = (s**0.5, cnt)
  return out


def build_degree_block_tridiag_M(A, deg_of_col, alpha_rel=1e-12, alpha_abs=1e-18):
  """
  Build dense block-tridiagonal approximation M ~ A^T A in degree blocks:
    M_dd   = (A^T A)_{dd}
    M_d,d+1 = (A^T A)_{d,d+1}

  Returns:
    M : (n,n) dense SPD-ish matrix (may require shift before Cholesky)
    blocks : list of (deg, J)
  """
  A = _as_csc(A)
  outdebug = degree_block_norms_of_normal(A, deg_of_col, max_delta=4)
  print(outdebug)
  blocks = _degree_blocks(deg_of_col)
  nb = len(blocks)
  n = A.shape[1]

  # Global permutation is identity because your columns are already in graded order.
  # We’ll assemble M in the original column order using index sets.
  M = np.zeros((n, n), dtype=np.float64)

  # Fill block diagonal and first off-diagonals
  for bi in range(nb):
    _, Jd = blocks[bi]
    H = _AtA_block(A, Jd, Jd)
    H = 0.5 * (H + H.T)

    # robust per-block regularization floor
    # scale by average diagonal magnitude
    diag_scale = float(np.mean(np.diag(H))) if H.size else 1.0
    reg = max(alpha_abs, alpha_rel * max(1.0, abs(diag_scale)))
    H = H + reg * np.eye(H.shape[0])

    M[np.ix_(Jd, Jd)] += H

    if bi + 1 < nb:
      _, Jp = blocks[bi + 1]
      K = _AtA_block(A, Jd, Jp)
      # symmetric placement
      M[np.ix_(Jd, Jp)] += K
      M[np.ix_(Jp, Jd)] += K.T

  return M, blocks

def build_degree_block_pentadiag_M_damped(
  A,
  deg_of_col,
  alpha_rel=1e-12,
  alpha_abs=1e-18,
  gamma_init=1.0,
  gamma_min=2.0**-20,
  gamma_shrink=0.5,
  eps_rel=1e-12,
):
  """
  Build a degree-band (|Δd|<=2) approximation M ~ A^T A that is SPD by construction.

  Strategy:
    - Build SPD block-diagonal D from exact normal blocks N_dd.
    - Build off-diagonals E1 (Δ=1), E2 (Δ=2).
    - Form M = D + gamma*(E1+E1^T + E2+E2^T).
    - Decrease gamma until chol_spd_global needs no (or tiny) shift.

  Returns:
    M      : dense (n,n)
    degs   : degrees present
    J_of_d : dict
    gamma  : final damping used
  """
  A = _as_csc(A)

  deg = np.asarray(deg_of_col, dtype=np.int64)
  nvar = A.shape[1]

  # Degree -> indices (only keep nonempty)
  J_of_d = {}
  for d in range(int(deg.min()), int(deg.max()) + 1):
    J = np.where(deg == d)[0]
    if J.size:
      J_of_d[int(d)] = J
  degs = sorted(J_of_d.keys())

  # Build block-diagonal D (SPD) and store off-diagonals separately
  D = np.zeros((nvar, nvar), dtype=np.float64)
  E = np.zeros((nvar, nvar), dtype=np.float64)  # will hold upper band (Δ=1,2)

  for d in degs:
    Jd = J_of_d[d]

    # N_dd
    H = _AtA_block(A, Jd, Jd)
    H = 0.5 * (H + H.T)

    diag_scale = float(np.mean(np.diag(H))) if H.size else 1.0
    reg = max(alpha_abs, alpha_rel * max(1.0, abs(diag_scale)))
    H = H + reg * np.eye(H.shape[0])

    D[np.ix_(Jd, Jd)] += H

    # Δ = 1 and 2 upper blocks into E
    for delta in (1, 2):
      dp = d + delta
      if dp not in J_of_d:
        continue
      Jp = J_of_d[dp]
      K = _AtA_block(A, Jd, Jp)
      E[np.ix_(Jd, Jp)] += K

  # Backtracking on gamma until Cholesky is happy without a big shift.
  gamma = float(gamma_init)
  last_shift = None

  while True:
    M = D + gamma * (E + E.T)

    # Try Cholesky with your global SPD helper
    _, shift = chol_spd_global(M, eps_rel=eps_rel)
    last_shift = shift

    # Accept if no shift (or extremely tiny shift)
    if shift == 0.0 or shift < 1e-10:
      break

    gamma *= float(gamma_shrink)
    if gamma < gamma_min:
      # Give up on off-diagonals; return block-diagonal (always SPD)
      gamma = 0.0
      M = D.copy()
      break

  if last_shift not in (0.0, None) and gamma == 0.0:
    print("Warning: could not make banded M SPD without shift; using block-diagonal only.")

  return M, degs, J_of_d, gamma


def chol_spd_global(M, eps_rel=1e-12):
  """
  Robust Cholesky of SPD matrix by applying a global diagonal shift if needed.
  Returns upper-triangular R such that M + shift*I = R^T R.
  """
  M = 0.5 * (M + M.T)

  # First try
  try:
    R = np.linalg.cholesky(M).T  # return upper
    return R, 0.0
  except np.linalg.LinAlgError:
    pass

  # Compute spectral bounds (dense, but this is for single-tet diagnostics)
  w = np.linalg.eigvalsh(M)
  wmin = float(w[0])
  wmax = float(w[-1])

  # shift so that min eigenvalue becomes eps_rel*wmax
  target = eps_rel * max(1.0, abs(wmax))
  shift = max(0.0, target - wmin)

  I = np.eye(M.shape[0])
  for _ in range(8):
    try:
      R = np.linalg.cholesky(M + shift * I).T
      return R, shift
    except np.linalg.LinAlgError:
      shift = max(1e-18, 10.0 * shift)

  raise np.linalg.LinAlgError("Failed to make M SPD even after shifting.")

def compute_all_singular_values_A_Minvhalf(
  A,
  deg_of_col,
  alpha_rel=1e-12,
  alpha_abs=1e-18,
  eps_rel=1e-12,
  band=2,
):
  """
  Compute all singular values of B = A * M^{-1/2}, where M is a dense
  degree-block banded approximation to A^T A.

  Parameters
  ----------
  A : (m,n) sparse or dense
    System matrix.
  deg_of_col : (n,) int
    Total degree per column (graded-lex ordering).
  alpha_rel, alpha_abs : float
    Per-block diagonal regularization used when assembling M.
  eps_rel : float
    Relative SPD shift target for Cholesky fallback.
  band : int
    Degree-bandwidth to keep in M:
      band=1 -> tridiagonal in degree (|d-d'|<=1)
      band=2 -> pentadiagonal in degree (|d-d'|<=2)

  Returns
  -------
  s : (n,) float
    All singular values of A * M^{-1/2}, sorted descending.
  shift : float
    Global diagonal shift applied (0 if no shift was needed).
  """
  A = _as_csc(A)
  m, n = A.shape

  if band == 1:
    M, _ = build_degree_block_tridiag_M(
      A, deg_of_col, alpha_rel=alpha_rel, alpha_abs=alpha_abs
    )
  elif band == 2:
    #M, _, _ = build_degree_block_pentadiag_M(A, deg_of_col, alpha_rel=alpha_rel, alpha_abs=alpha_abs)
    #M, *_ = build_degree_block_pentadiag_M_coarse01(A, deg_of_col, alpha_rel=alpha_rel, alpha_abs=alpha_abs)
    #M, _, _ = build_degree_block_pentadiag_M_additive_coarse01(A, deg_of_col, alpha_rel=alpha_rel, alpha_abs=alpha_abs)
    M, _, _, gamma = build_degree_block_pentadiag_M_damped(
      A, deg_of_col,
      alpha_rel=alpha_rel,
      alpha_abs=alpha_abs,
      eps_rel=eps_rel,
    )
    print("gamma used =", gamma)

  else:
    raise ValueError(f"Unsupported band={band}. Use band=1 or band=2.")

  R, shift = chol_spd_global(M, eps_rel=eps_rel)

  # X = R^{-1} by triangular solves against I
  I = np.eye(n)
  X = scipy.linalg.solve_triangular(R, I, lower=False, check_finite=False)
  #plt.spy(np.abs(R)>1e-10)
  #plt.show()
  # Dense A
  Ad = A.toarray() if sps.issparse(A) else np.asarray(A)
  B = Ad @ X

  s = np.linalg.svd(B, compute_uv=False)
  return s, shift



def _chol_spd_with_shift(A, eps_rel=1e-12, max_tries=6):
  """
  Return lower-triangular L such that (A + shift*I) = L L^T is SPD.
  We symmetrize A and add the smallest shift needed based on min eigenvalue.
  """
  A = 0.5 * (A + A.T)

  # Quick attempt first
  try:
    return np.linalg.cholesky(A)
  except np.linalg.LinAlgError:
    pass

  # Compute minimum eigenvalue (block sizes are small enough)
  wmin = float(np.min(np.linalg.eigvalsh(A)))
  # Target: wmin + shift >= eps_rel * ||A||_2
  # Use spectral radius approximation via max eigenvalue
  wmax = float(np.max(np.linalg.eigvalsh(A)))
  target = eps_rel * max(1.0, abs(wmax))
  shift = max(0.0, target - wmin)

  I = np.eye(A.shape[0])
  for _ in range(max_tries):
    try:
      return np.linalg.cholesky(A + shift * I)
    except np.linalg.LinAlgError:
      shift *= 10.0 if shift > 0 else 1e-12

  # If we get here, something is seriously indefinite / ill-scaled.
  # Fall back to a larger shift.
  shift = max(1e-8, 1e-6 * max(1.0, abs(wmax)))
  return np.linalg.cholesky(A + shift * I)


def plot_spectrum(s, title="Singular values of preconditioned operator"):
  """
  Plot singular values on log-log and semi-log.
  """
  s = np.asarray(s, dtype=np.float64)
  idx = np.arange(1, s.size + 1)
  plt.figure()
  plt.semilogy(idx, s, marker=".", linewidth=1.0)
  plt.xlabel("index")
  plt.ylabel("singular value")
  plt.title(title + " (semi-log)") 
  plt.grid(True, which="both", ls="--", alpha=0.4)
  
  plt.show()



class RefTetPrecomp:
  def __init__(self, n, q_vol, q_face, kappa_src):
    self.D = 3
    self.n = int(n)
    self.q_vol = int(q_vol)
    self.q_face = int(q_face)
    self.kappa_src = np.asarray(kappa_src, dtype=np.float64)
    self.face_sigma = [(0,1,2), (0,1,2), (0,1,2), (0,1,2)] 
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
    self.tau_ref = np.linalg.norm(self.L_ref, ord=2)**2 / np.linalg.norm(self.T_ref, ord=2)**2
    self.R = self._precompute_precond()

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
    return R


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
    self.L_int = self.detJabs * A_lap[:ref.m, :]
    self.T_full, self.F_full = self._assemble_TF_full_sigma()

    self.int_fact = interior_qr_factor(self.L_int, rtol=rtol_int)
    self.N = self.int_fact["N"]  # (M,k)


    fro, stats = nnz_stats(self.T_full)
    #print("[T_full] ||.||_F =", fro)
    #for thr, nnz, frac in stats:
    #  print(f"[T_full] nnz(frac) for abs/||.||_F > {thr:g}: {nnz} ({frac:.6e})")

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
  
  def _assemble_TF_full_sigma(self):
    ref = self.ref
    T_blocks = []
    F_blocks = []
  
    for face_id in range(4):
      sigma = self.face_sigma[face_id]
      fd = ref.face[face_id]
  
      # reference precomputed sparse blocks for this face+sigma
      Tf_ref = fd["T_sigma_ref"][sigma]  # (mt_face, Mvol)
      Ff_ref = fd["F_sigma_ref"][sigma]  # (mt_face, Mvol)
  
      # geometry scalars
      n_hat = fd["n_hat"]
      n_tilde = self.JinvT @ n_hat
      n_tilde_norm = float(np.linalg.norm(n_tilde))
  
      # matches your original scaling:
      # wS_phys = wS_hat * detJabs * ||n_tilde||
      # wF_phys = wS_hat * detJabs
      sT = float(self.detJabs * n_tilde_norm)
      sF = float(self.detJabs)
  
      # sparse scaling (preserves sparsity)
      T_blocks.append(sT * Tf_ref)
      F_blocks.append(sF * Ff_ref)
  
    # sparse vstack
    #T_full = sps.vstack(T_blocks, format="csc")
    #F_full = sps.vstack(F_blocks, format="csc")
    T_full = np.vstack(T_blocks)
    F_full = np.vstack(F_blocks)
    return T_full, F_full

  #def _assemble_TF_full_sigma(self):
  #  ref = self.ref
  #  T_blocks = []
  #  F_blocks = []
  #  for face_id in range(4):
  #    sigma = self.face_sigma[face_id]
  #    fd = ref.face[face_id]

  #    wS_hat = fd["wS_hat"]
  #    Vt = fd["Vt"]
  #    n_hat = fd["n_hat"]

  #    Vv = fd["Vv_sigma"][sigma]
  #    dVv_hat = fd["dVv_hat_sigma"][sigma]

  #    n_tilde = self.JinvT @ n_hat
  #    n_tilde_norm = float(np.linalg.norm(n_tilde))

  #    wS_phys = wS_hat * (self.detJabs * n_tilde_norm)
  #    wF_phys = wS_hat * self.detJabs

  #    Tf = Vt.T @ (wS_phys[:, None] * Vv)

  #    dVv_x = np.einsum("ab,qmb->qma", self.JinvT, dVv_hat)
  #    ndot = (n_tilde[0] * dVv_x[:, :, 0] +
  #            n_tilde[1] * dVv_x[:, :, 1] +
  #            n_tilde[2] * dVv_x[:, :, 2])
  #    Ff = Vt.T @ (wF_phys[:, None] * ndot)

  #    T_blocks.append(Tf)
  #    F_blocks.append(Ff)

  #  return np.vstack(T_blocks), np.vstack(F_blocks)

  def project_f_int(self, f_rhs_phys):
    Xhat = self.ref.Xhat_vol_lap
    w = self.ref.w_vol_lap
    V = self.ref.V_vol_lap
    X = map_ref_to_phys(Xhat, self.J, self.b)
    fv = f_rhs_phys(X[:, 0], X[:, 1], X[:, 2])
    f_lap = V.T @ (w * fv)
    return self.detJabs * f_lap[:self.ref.m]


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
  u_expr = 0
  for a in range(m_max + 1):
    for b in range(m_max + 1 - a):
      for c in range(m_max + 1 - a - b):
        coef = sp.Rational(1, 1 + a + b + c)
        u_expr += coef * (x**a) * (y**b) * (z**c)
  #u_expr = sp.exp(x**2 + y**2 + z**2)

  return make_sym_u_f_and_grad(u_expr, simplify=True)  

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
  
  #Tproj = Ur.T @ leaf.T_full
  #lam_proj = Ur.T @ lam_full

  #A = np.vstack([
  #  w_bc * Tproj,
  #  w_pde * leaf.L_int,
  #])
  #b = np.concatenate([
  #  w_bc * lam_proj,
  #  -w_pde * f_int,
  #])



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
  A[np.abs(A)/np.linalg.norm(A,'fro') < 1e-14] = 0.0
  #plt.spy(A)
  #plt.show()
  c, resid, rnk, s = scipy.linalg.lstsq(A, b, lapack_driver="gelsd", check_finite=False)
  deg_of_col = leaf.ref.alpha_src.sum(axis=1)
  #s = compute_all_preconditioned_singular_values_dense(
  #  A, deg_of_col,
  #  alpha=1e-14
  #)
  #tau = leaf.compute_tau_geom(tau_ref=leaf.ref.tau_ref)#
  #tau = np.linalg.norm(L, ord=2)**2 / np.linalg.norm(T, ord=2)**2
  tau = leaf.ref.tau_ref
  A = np.vstack([
    np.sqrt(tau)*T,
    L,
  ])
  tau_phys = (np.linalg.norm(L, ord=2) / (np.linalg.norm(T, ord=2) + 1e-300))**2
  print("tau_ref =", tau, "tau_phys =", tau_phys, "ratio =", tau_phys/(tau + 1e-300))
 

  s_pc, shift = compute_all_singular_values_A_Minvhalf(A, deg_of_col, band=2)
  print("chol_shift =", shift, "smax =", s[0], "smin =", s[-1], "precond =" , s_pc[0]/s_pc[-1])

  deg = deg_of_col
  assert deg.shape[0] == A.shape[1]
  assert np.all(deg[:-1] <= deg[1:]) 
  Ad = A.toarray() if sps.issparse(A) else np.asarray(A)
  assert np.isfinite(Ad).all()
  plot_spectrum(s_pc, title="svd(A @ M^{-1/2})")


  #plt.semilogy(np.abs(s))
  #plt.show()
  # Diagnostics
  bc_res = Ur.T @ (T @ c) - Ur.T @ lam_full#T @ c - lam_full
  pde_res = L @ c + np.asarray(f_int, dtype=np.float64)

  rel_bc = float(np.linalg.norm(bc_res) / (np.linalg.norm(Ur.T @ lam_full) + 1e-300))
  rel_pde = float(np.linalg.norm(pde_res) / (np.linalg.norm(f_int) + 1e-300))
  # How much of lam_full is outside Range(T N)
  lam_rep = Ur @ (Ur.T @ lam_full)
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

  tetA_g = (0, 1, 2, 3)
  #tetA_g = (10, 3, 5, 57)
  sigA = build_face_sigma_list_for_tet(tetA_g)

  q_err_min = int(m_max + q_pad)

  dofs = []
  errs = []

  for n in range(12, n_max + 1):
    q_vol = n + q_pad
    q_face = n + q_pad
    ref = RefTetPrecomp(n=n, q_vol=q_vol, q_face=q_face, kappa_src=kappa_src)
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
  dofs, errs = run_single_tet_poly_convergence(m_max=12, n_max=14, q_pad=1, do_print=True, method="minls")
  s = np.zeros_like(errs)
  for j in range(len(errs)):
    if j == 0:
      s[j] = np.log(errs[j])
    else:
      s[j] = np.log(errs[j])-np.log(errs[j-1])
  plt.semilogy(dofs, errs)
  plt.show()
