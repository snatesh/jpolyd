import numpy as np
from jmat import *
from jbasis import *
from jquad_tprod import *
import matplotlib.pyplot as plt
import scipy.sparse as sp

def comb(n, k):
  if k < 0 or k > n:
    return 0
  k = min(k, n - k)
  num = 1
  den = 1
  for i in range(1, k + 1):
    num *= n - (k - i)
    den *= i
  return num // den

def dim_R(D, n):
  # dim of homogeneous polynomials of degree n in D variables:
  # C(n + D - 1, n)
  return comb(n + D - 1, n)

def degree_offsets(D, n):
  # off[j] = start index of degree-j block in the total-degree ordering
  off = [0]
  for j in range(1, n + 1):
    off.append(off[-1] + dim_R(D, j - 1))
  # off has length n+1, with off[n] = start of degree-n block
  return off

def degree_slice(off, D, j):
  a = off[j]
  b = a + dim_R(D, j)
  return slice(a, b)

def split_by_degree(D, n, v):
  off = degree_offsets(D, n)
  blocks = []
  for j in range(n + 1):
    sj = degree_slice(off, D, j)
    blocks.append(v[sj].copy())
  return blocks

def join_by_degree(D, n, blocks):
  off = degree_offsets(D, n)
  M = off[-1] + dim_R(D, n)
  v = np.zeros(M, dtype=np.float64)
  for j in range(n + 1):
    sj = degree_slice(off, D, j)
    v[sj] = blocks[j]
  return v

def build_basis_structs(D, n, kappa):
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)
  M = alpha_table.shape[0]
  return alpha_table, tail_deg, inv_h, M

def eval_poly_x(X, poly_terms):
  # X: (npts, D) points in simplex coords, return q(X)
  q = np.zeros(X.shape[0], dtype=np.float64)
  for beta, a in poly_terms.items():
    term = np.ones(X.shape[0], dtype=np.float64)
    for i, bi in enumerate(beta):
      if bi:
        term *= X[:, i] ** int(bi)
    q += a * term
  return q

def beta(*exps, D=None):
  if D is None:
    raise ValueError("beta(..., D=) required")
  exps = list(exps)
  if len(exps) > D:
    raise ValueError(f"beta length {len(exps)} > D={D}")
  return tuple(exps + [0]*(D - len(exps)))

def dim_Pi(D, n):
  # dim total-degree <= n
  # sum_{j=0}^n C(j + D - 1, j) = C(n + D, D)
  # use integer comb from your file
  return comb(n + D, D)

def embed_to_big(c_small, D, n, n_big):
  M_small = dim_Pi(D, n)
  M_big = dim_Pi(D, n_big)
  c_big = np.zeros(M_big, dtype=np.float64)
  c_big[:M_small] = c_small
  return c_big

def restrict_to_small(c_big, D, n):
  M_small = dim_Pi(D, n)
  return c_big[:M_small].copy()

def apply_monomial_J_big(J_big, c_big, beta):
  # beta length = D
  y = c_big
  for i, bi in enumerate(beta):
    for _ in range(int(bi)):
      y = J_big[i] @ y
  return y

def apply_poly_J_big(J_big, c_big, poly_terms):
  y = np.zeros_like(c_big)
  for beta, a in poly_terms.items():
    y += a * apply_monomial_J_big(J_big, c_big, beta)
  return y

def baseline_projected_poly(D, n, kappa, c_small, poly_terms):
  # p = max total degree of q
  p = max(sum(beta) for beta in poly_terms.keys())
  n_big = n + p

  J_big = jmat_build_csc(D, n_big, kappa)

  c_big = embed_to_big(c_small, D, n, n_big)
  y_big = apply_poly_J_big(J_big, c_big, poly_terms)
  y_small = restrict_to_small(y_big, D, n)
  return y_small

def baseline_multbyq_from_basis(V, W, c, qcoef):
  """
  Baseline: y = Pi_n(q * f) using quadrature and discrete orthonormality.
    f(x) = V c
    q(x) = V qcoef
    ycoef = V^T W (q(x) f(x))
  """
  u = V @ c
  qx = V @ qcoef
  ycoef = V.T @ (W * (qx * u))
  return ycoef

def extract_U_blocks_from_J(J_all, D, n):
  """
  Extract U_{i,j} = E_{j+1}^T J_i E_j from CSC matrices, as dense arrays for testing.
  Returns U[i][j] with shapes (m_{j+1}, m_j).
  """
  off = degree_offsets(D, n)
  U = [[None]*(n) for _ in range(D)]  # j=0..n-1
  for i in range(D):
    Ji = J_all[i]
    for j in range(n):
      sj  = degree_slice(off, D, j)
      sj1 = degree_slice(off, D, j+1)
      # (j+1 rows) x (j cols)
      U[i][j] = Ji[sj1, sj].toarray()
  return U

def precompute_Uplus(U, D, n):
  """
  Build stacked U_j and its left inverse U_j^+ via least squares/pinv for testing.
  Returns Uplus[j] shape (m_j, D*m_{j+1}) for j=0..n-1.
  """
  Uplus = [None]*n
  for j in range(n):
    if j == n:
      break
  for j in range(n):
    if j == n:
      continue
    # stack vertically
    Ustack = np.vstack([U[i][j] for i in range(D)])  # (D*m_{j+1}, m_j)
    # left inverse (min-norm): U^+ = (U^T U)^{-1} U^T
    G = Ustack.T @ Ustack
    Uplus[j] = np.linalg.solve(G, Ustack.T)
  return Uplus

def extract_Uij(Ji, off, D, j):
  # U_{i,j} = E_{j+1}^T J_i E_j
  sj  = degree_slice(off, D, j)
  sj1 = degree_slice(off, D, j+1)
  return Ji[sj1, sj].toarray()

def check_Uplus_identity(J_all, D, n, trials=5, seed=0):
  off = degree_offsets(D, n)
  rng = np.random.default_rng(seed)

  for j in range(0, n):
    mj  = dim_R(D, j)
    mj1 = dim_R(D, j+1)

    Ustack = np.vstack([extract_Uij(J_all[i], off, D, j) for i in range(D)])  # (D*mj1, mj)

    # left inverse via normal eq for testing
    G = Ustack.T @ Ustack
    Uplus = np.linalg.solve(G, Ustack.T)  # (mj, D*mj1)

    # test Uplus*Ustack ≈ I
    Ierr = np.linalg.norm(Uplus @ Ustack - np.eye(mj)) / np.linalg.norm(np.eye(mj))
    # random range test: Uplus(Ustack v) ≈ v
    rerr = 0.0
    for _ in range(trials):
      v = rng.standard_normal(mj)
      vv = Uplus @ (Ustack @ v)
      rerr = max(rerr, np.linalg.norm(vv - v) / (np.linalg.norm(v) + 1e-300))

    print(f"[Uplus check] j={j:2d}  rel(||U+U-I||_F)={Ierr:.3e}  max_rel(U+U v - v)={rerr:.3e}")

def clenshaw_apply_multbyq(J_all, D, n, q_blocks, c, Uplus):
  """
  Prototype operator-valued Clenshaw-style apply for y = M_q c on Pi_n^D.

  q_blocks: list length n+1, q_blocks[j] in R^{m_j}.
  Uplus[j]: left inverse for stacked raising operator at level j (j=0..n-1).
  """
  off = degree_offsets(D, n)
  M = off[-1] + dim_R(D, n)

  # embed q as global vectors Q_j = E_j q_j
  Q = [np.zeros(M, dtype=np.float64) for _ in range(n+1)]
  for j in range(n+1):
    sj = degree_slice(off, D, j)
    Q[j][sj] = q_blocks[j]

  # backward states
  bjp2 = np.zeros(M, dtype=np.float64)  # b_{j+2}
  bjp1 = np.zeros(M, dtype=np.float64)  # b_{j+1}

  # go down j = n..0 (you can also do p..0 if q has smaller degree)
  for j in range(n, -1, -1):
    # forcing term: degree-j coefficients of q multiplied against c in coefficient space
    # (this is consistent with your earlier exposition; keep it for now)
    bj = Q[j] * c

    if j <= n - 1:
      # form z_i = J_i b_{j+1}
      Z = []
      for i in range(D):
        zi = J_all[i] @ bjp1
        sj1 = degree_slice(off, D, j+1)
        Z.append(zi[sj1])
      Zstack = np.concatenate(Z, axis=0)  # (D*m_{j+1},)

      # fold down to degree-j
      tj = Uplus[j] @ Zstack
      sj = degree_slice(off, D, j)
      bj[sj] += tj

    # simple down-correction via b_{j+2} (prototype; will refine as needed)
    if j <= n - 2:
      # subtract the degree-j part coming from J_i acting on degree-(j+2) of b_{j+2}
      # which is exactly U_{i,j+1}^T * b_{j+2}^{(j+2)} when J_i is symmetric.
      sj2 = degree_slice(off, D, j+2)
      vjp2 = bjp2[sj2]
      sj = degree_slice(off, D, j)
      corr = np.zeros(dim_R(D, j), dtype=np.float64)
      for i in range(D):
        # pull (j, j+1) block from Ji and multiply by (j+2) block via (j+1,j+2) then fold?
        # keep prototype minimal: use full Ji multiply then extract degree j
        corr += (J_all[i] @ bjp2)[sj]
      bj[sj] -= corr

    # shift
    bjp2, bjp1 = bjp1, bj

  return bjp1  # b_0

def assemble_Mq_dense(V, W, qcoef):
  qx = V @ qcoef                  # (npts,)
  A = (W * qx)[:, None] * V       # (npts, M)
  return V.T @ A  

def test_multbyq_coordinate_poly(D=3, n=8, kappa=None, nquad=None, seed=0):
  if kappa is None:
    kappa = np.array([0.5] * (D + 1), dtype=np.float64)
  if nquad is None:
    nquad = n + 2  # usually safe for these tests

  # build J_i (CSC)
  J_all = jmat_build_csc(D, n, kappa)

  # basis structs + evaluation matrix
  alpha_table, tail_deg, inv_h, M = build_basis_structs(D, n, kappa)
  X, W = jquad_mapped_build_kappa(D, nquad, kappa)
  V = jbasis_eval_all(X, kappa, n, alpha_table, tail_deg, inv_h, D)  # (npts, M)

  rng = np.random.default_rng(seed)
  c = rng.standard_normal(M)

  # pick a small polynomial q(x) in coordinates
  # example: q = 0.7 + 0.2 x1 - 0.4 x2^2 + 0.1 x1 x3
  poly_terms = {
    beta(0, D=D): 0.7,
    beta(1, D=D): 0.2,       # x1
  }
  
  if D >= 2:
    poly_terms[beta(0,2, D=D)] = -0.4   # x2^2
  if D >= 3:
    poly_terms[beta(1,0,1, D=D)] = 0.1  # x1*x3

  # coefficient-space: y = q(J)c
  y = baseline_projected_poly(D, n, kappa, c, poly_terms)

  # physical-space compare
  u = V @ c
  qx = eval_poly_x(X, poly_terms)
  # project product back to Pi_n so we can compare with 
 
  c_prod = V.T @ (W * (qx * u))
  rcoef = y - c_prod
  rel_c = np.linalg.norm(rcoef) / (np.linalg.norm(c_prod) + 1e-300)
  # compare in physical space after projection
  lhs = V @ y
  rhs = V @ c_prod
  r = lhs - rhs
  rel_L2 = np.sqrt(np.sum(W * (r * r))) / (np.sqrt(np.sum(W * (rhs * rhs))) + 1e-300)

  print(f"[multbyq baseline] D={D} n={n} nquad={nquad} rel_coef={rel_c:.3e} rel_L2kappa={rel_L2:.3e}")
  return rel_c

def test_clenshaw_multbyq(D=2, n=8, kappa=None, nquad=None, seed=0):
  if kappa is None:
    kappa = np.array([0.5]*(D+1), dtype=np.float64)
  if nquad is None:
    nquad = n + 2

  # J and basis eval
  J_all = jmat_build_csc(D, n, kappa)
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)
  X, W = jquad_mapped_build_kappa(D, nquad, kappa)
  V = jbasis_eval_all(X, kappa, n, alpha_table, tail_deg, inv_h, D)

  M = V.shape[1]
  rng = np.random.default_rng(seed)
  c = rng.standard_normal(M)
  qcoef = rng.standard_normal(M)

  # q blocks
  q_blocks = split_by_degree(D, n, qcoef)

  # Uplus precompute
  U = extract_U_blocks_from_J(J_all, D, n)
  Uplus = precompute_Uplus(U, D, n)

  # baseline
  y_ref = baseline_multbyq_from_basis(V, W, c, qcoef)

  # clenshaw (prototype)
  y_cl = clenshaw_apply_multbyq(J_all, D, n, q_blocks, c, Uplus)

  r = y_cl - y_ref
  rel = np.linalg.norm(r) / (np.linalg.norm(y_ref) + 1e-300)
  print(f"[clenshaw test] D={D} n={n} nquad={nquad} rel_coef={rel:.3e}")
  return rel

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def degree_slices(deg_offsets, j):
  a = deg_offsets[j]
  b = deg_offsets[j + 1]
  return slice(a, b)

def extract_block(A, row_slc, col_slc):
  # A is sparse; slice returns sparse
  return A[row_slc, col_slc].tocsc()

def pinv_left(B, rcond=1e-14):
  # Dense left pseudoinverse for testing.
  # B: (D*mjp1) x (mj)
  Bd = B.toarray() if sp.issparse(B) else np.asarray(B)
  # Moore-Penrose via SVD
  U, s, Vt = np.linalg.svd(Bd, full_matrices=False)
  s_inv = np.zeros_like(s)
  tol = rcond * s.max() if s.size else 0.0
  s_inv[s > tol] = 1.0 / s[s > tol]
  return (Vt.T * s_inv) @ U.T  # (mj) x (D*mjp1)

def build_Aj_Bj(J_all, deg_offsets, D, j, Bplus_cache, C_cache):
  """
  Build sparse matrices A_j and B_j (both MN x MN) such that:
    b_j - A_j b_{j+1} + B_j b_{j+2} = rhs_j
  """
  MN = J_all[0].shape[0]
  sj = degree_slices(deg_offsets, j)
  sj1 = degree_slices(deg_offsets, j + 1)
  sj2 = degree_slices(deg_offsets, j + 2)

  mj = sj.stop - sj.start
  mj1 = sj1.stop - sj1.start

  # ---- Build B_j (stacked raising block) and its dense left pseudoinverse Bplus_j
  if j not in Bplus_cache:
    U_blocks = []
    for i in range(D):
      # U_{i,j} = E_{j+1}^T J_i E_j  (rows j+1, cols j)
      Uij = extract_block(J_all[i], sj1, sj)
      U_blocks.append(Uij)
    Bj = sp.vstack(U_blocks, format="csc")  # (D*mj1) x mj
    Bplus_cache[j] = pinv_left(Bj)          # dense (mj) x (D*mj1)

  Bplus_j = Bplus_cache[j]

  # ---- Build stacked extractor S_{j+1}: w -> [E_{j+1}^T(J_1 w); ...; E_{j+1}^T(J_D w)]
  # As a matrix: S_{j+1} is (D*mj1) x MN obtained by stacking (E_{j+1}^T J_i).
  S_blocks = []
  for i in range(D):
    S_i = J_all[i][sj1, :]   # (mj1) x MN
    S_blocks.append(S_i)
  Sjp1 = sp.vstack(S_blocks, format="csc")  # (D*mj1) x MN

  # ---- A_j acts: w -> embed_j( Bplus_j @ (S_{j+1} w) )
  # Matrix form: A_j = E_j * Bplus_j * S_{j+1}
  # Build as sparse by forming (mj x MN) dense*sparse then embedding.
  A_mid = Bplus_j @ Sjp1.toarray()  # TEST PATH: dense. Replace later with operator application.
  Aj = sp.csc_matrix((MN, MN))
  Aj = Aj + sp.csc_matrix((A_mid.shape[0], A_mid.shape[1]))  # placeholder

  # Embed only into degree-j rows:
  # Aj[sj, :] = A_mid
  Aj = sp.lil_matrix((MN, MN))
  Aj[sj, :] = A_mid
  Aj = Aj.tocsc()

  # ---- Build correction C_{j+1} = Bplus_j * [B_{1,j+1}; ...; B_{D,j+1}]
  if j not in C_cache:
    Bsame_blocks = []
    for i in range(D):
      Bij1 = extract_block(J_all[i], sj1, sj1)  # B_{i,j+1}
      Bsame_blocks.append(Bij1)
    Bstack = sp.vstack(Bsame_blocks, format="csc")  # (D*mj1) x mj1
    Cj1 = Bplus_j @ Bstack.toarray()                # dense (mj) x (mj1), TEST PATH
    C_cache[j] = Cj1
  Cj1 = C_cache[j]

  # B_j operator in the two-step recurrence is:
  #   embed_j( C_{j+1} * extract_{j+1}(w) )
  Bj2 = sp.lil_matrix((MN, MN))
  Bj2[sj, sj1] = Cj1
  Bj2 = Bj2.tocsc()

  return Aj, Bj2

def build_LpT(J_all, deg_offsets, D, p):
  """
  Build the block upper-triangular matrix LpT = \tilde L_p(J)^T of size ((p+1)MN) x ((p+1)MN)
  for the 2-step recurrence:
    b_j - A_j b_{j+1} + B_j b_{j+2} = rhs_j.
  """
  MN = J_all[0].shape[0]
  Bplus_cache = {}
  C_cache = {}

  blocks = [[None for _ in range(p + 1)] for _ in range(p + 1)]

  I = sp.eye(MN, format="csc")

  for j in range(p + 1):
    blocks[j][j] = I

    if j + 1 <= p:
      Aj, _ = build_Aj_Bj(J_all, deg_offsets, D, j, Bplus_cache, C_cache)
      blocks[j][j + 1] = -Aj

    if j + 2 <= p:
      _, Bj2 = build_Aj_Bj(J_all, deg_offsets, D, j, Bplus_cache, C_cache)
      blocks[j][j + 2] = Bj2

  # Fill unset with zeros
  Z = sp.csc_matrix((MN, MN))
  for r in range(p + 1):
    for c in range(p + 1):
      if blocks[r][c] is None:
        blocks[r][c] = Z

  LpT = sp.bmat(blocks, format="csc")
  return LpT




test_multbyq_coordinate_poly(D=1, n=12, kappa=np.array([0.8, 1.7]))
test_multbyq_coordinate_poly(D=2, n=10, kappa=np.array([0.8, 1.7, 2.3]))
test_multbyq_coordinate_poly(D=3, n= 8, kappa=np.array([1.7, 3.3, 2.8, 0.9]))
#test_clenshaw_multbyq(D=2, n=8, kappa=[0.8, 1.7, 2.3])
J_all = jmat_build_csc(D=2, n=8, kappa=np.array([0.8,1.7,2.3]))
check_Uplus_identity(J_all, D=2, n=8)

