import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from jmat import *
from jbasis import *
from jquad_tprod import *

# --------------------------
# Small utilities
# --------------------------

def build_basis_structs(D, n, kappa):
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)
  M = alpha_table.shape[0]
  return alpha_table, tail_deg, inv_h, M

def degree_offsets_from_alpha(alpha_table, p):
  deg = np.sum(alpha_table, axis=1).astype(np.int64)
  M = alpha_table.shape[0]
  off = np.zeros(p + 2, dtype=np.int64)
  k = 0
  for j in range(p + 1):
    off[j] = k
    while k < M and deg[k] == j:
      k += 1
  off[p + 1] = k
  if off[p + 1] != M:
    raise ValueError(f"alpha_table not truncated to p={p}? off[p+1]={off[p+1]} M={M}")
  return off

def comb_int(n, k):
  if k < 0 or k > n:
    return 0
  k = min(k, n - k)
  num = 1
  den = 1
  for i in range(1, k + 1):
    num *= (n - k + i)
    den *= i
  return num // den

def m_hom(D, j):
  # dim homogeneous degree-j in D vars
  return comb_int(j + D - 1, D - 1)

def _add_A_kron_I(rows, cols, vals, A, row0, col0, Mfun, scale=1.0):
  if A.nnz == 0:
    return
  Acoo = A.tocoo()
  rr, cc, vv = Acoo.row, Acoo.col, Acoo.data
  for k in range(vv.shape[0]):
    r = int(rr[k])
    c = int(cc[k])
    a = scale * vv[k]
    br = row0 + r * Mfun
    bc = col0 + c * Mfun
    for t in range(Mfun):
      rows.append(br + t)
      cols.append(bc + t)
      vals.append(a)

# --------------------------
# Canonical pivot rule for general D
# --------------------------

def pivot_rows_D_canonical(alpha_table_p, off_p, j, D):
  """
  Return pivot rows in stacked coordinate-row space of size (D*mj),
  selecting exactly m_{j+1} rows.

  Blocks are ordered i=0..D-1, each of size mj corresponding to coordinate x_{i+1}.
  Selection rule:
    - block 0: take all mj rows
    - block 1: take rows where alpha_0 == 0
    - block 2: take rows where alpha_0 == alpha_1 == 0
    ...
    - block i: take rows where alpha_0..alpha_{i-1} == 0
  """
  s0 = int(off_p[j])
  s1 = int(off_p[j + 1])
  mj = s1 - s0
  alpha_j = alpha_table_p[s0:s1, :]  # (mj, D)

  piv = []

  # block 0: all rows
  piv.extend(list(range(0, mj)))

  # blocks 1..D-1: prefix zeros
  for i in range(1, D):
    # select local indices where alpha_0..alpha_{i-1} == 0
    prefix = alpha_j[:, :i]
    sel = np.where(np.all(prefix == 0, axis=1))[0].astype(np.int64)
    piv.extend((i * mj + sel).tolist())

  piv = np.array(piv, dtype=np.int64)

  mj1 = m_hom(D, j + 1)
  if piv.size != mj1:
    raise ValueError(f"pivot size mismatch at j={j}: got {piv.size}, expected {mj1}")
  return piv

# --------------------------
# Build square L for general D, poly degree p, function degree K
# --------------------------

def build_L_otv_square_D(p, K, D, kappa, assume_symmetric=True):
  """
  Build square OTV-style L for general D using canonical pivot rule.

  - J_poly is built at degree p (Mp x Mp), used to extract A,B,C blocks per coord.
  - J_fun  is built at degree K (MK x MK), used inside row-picked -(I ⊗ J_i).

  Returns:
    L  : CSC, shape (Mp*MK, Mp*MK)
    meta: dict with offsets
  """
  # poly structs
  alpha_p, tail_p, invh_p, Mp = build_basis_structs(D, p, kappa)
  off_p = degree_offsets_from_alpha(alpha_p, p)

  # fun structs
  alpha_K, tail_K, invh_K, MK = build_basis_structs(D, K, kappa)

  # Jacobi matrices (CSC)
  J_poly = jmat_build_csc(D, p, kappa)  # list length D, each Mp x Mp
  J_fun  = jmat_build_csc(D, K, kappa)  # list length D, each MK x MK

  for i in range(D):
    if J_poly[i].shape != (Mp, Mp):
      raise ValueError("J_poly shape mismatch")
    if J_fun[i].shape != (MK, MK):
      raise ValueError("J_fun shape mismatch")

  # unknown length n = Mp * MK
  n = Mp * MK

  # block offsets for v_j in lifted space: v_j size m_j * MK
  blk_off = np.zeros(p + 2, dtype=np.int64)
  acc = 0
  for j in range(p + 1):
    mj = m_hom(D, j)
    blk_off[j] = acc
    acc += mj * MK
  blk_off[p + 1] = acc
  if acc != n:
    raise ValueError(f"block sum mismatch acc={acc} n={n}")

  rows, cols, vals = [], [], []

  # anchor: I_MK in (v0,v0) (m0=1)
  for t in range(MK):
    rows.append(t)
    cols.append(t)
    vals.append(1.0)

  # COO copies for fast placement of J_fun blocks
  Jfun_coo = [J_fun[i].tocoo() for i in range(D)]

  row_cursor = MK

  for j in range(p):
    mj   = m_hom(D, j)
    mj1  = m_hom(D, j + 1)
    mjm1 = m_hom(D, j - 1) if j >= 1 else 0

    # poly slices
    sj0, sj1 = int(off_p[j]), int(off_p[j + 1])
    sjp10, sjp11 = int(off_p[j + 1]), int(off_p[j + 2])

    # Build stacked A,B,C across coordinates
    A_blocks = []
    B_blocks = []
    C_blocks = [] if j >= 1 else None

    for i in range(D):
      Ji_p = J_poly[i].tocsc()
      A_blocks.append(Ji_p[sj0:sj1, sj0:sj1].tocsc())          # (mj x mj)
      B_blocks.append(Ji_p[sj0:sj1, sjp10:sjp11].tocsc())      # (mj x mj1)

      if j >= 1:
        sm10, sm11 = int(off_p[j - 1]), int(off_p[j])
        if assume_symmetric:
          # (j-1, j) block is (mjm1 x mj); transpose to (mj x mjm1)
          Bj_m1 = Ji_p[sm10:sm11, sj0:sj1].tocsc()
          C_blocks.append(Bj_m1.T.tocsc())
        else:
          C_blocks.append(Ji_p[sj0:sj1, sm10:sm11].tocsc())

    Astack = sp.vstack(A_blocks, format="csc")  # (D*mj) x mj
    Bstack = sp.vstack(B_blocks, format="csc")  # (D*mj) x mj1
    Cstack = sp.vstack(C_blocks, format="csc") if j >= 1 else None  # (D*mj) x mjm1

    # canonical pivot rows length mj1
    piv = pivot_rows_D_canonical(alpha_p, off_p, j, D)

    # row-picked effective blocks
    Aeff = Astack[piv, :].tocsc()              # (mj1 x mj)
    Beff = Bstack[piv, :].tocsc()              # (mj1 x mj1)
    Ceff = Cstack[piv, :].tocsc() if j >= 1 else None  # (mj1 x mjm1)

    # global row offset for this row-block
    r0 = row_cursor
    row_cursor += mj1 * MK

    # column offsets
    c_jm1 = int(blk_off[j - 1]) if j >= 1 else None
    c_j   = int(blk_off[j])
    c_jp1 = int(blk_off[j + 1])

    # left: (Ceff ⊗ I)
    if j >= 1:
      _add_A_kron_I(rows, cols, vals, Ceff, r0, c_jm1, MK, scale=+1.0)

    # middle: (Aeff ⊗ I)
    _add_A_kron_I(rows, cols, vals, Aeff, r0, c_j, MK, scale=+1.0)

    # row-picked -(I ⊗ J_fun_i) for selected stacked rows
    # r_stack in [0, D*mj): coord i = r_stack // mj, local b = r_stack % mj
    for r_sel in range(mj1):
      r_stack = int(piv[r_sel])
      i = r_stack // mj
      b = r_stack - i * mj

      Jcoo = Jfun_coo[i]
      row_block = r0 + r_sel * MK
      col_block = c_j + b * MK

      for k in range(Jcoo.data.shape[0]):
        rr = int(Jcoo.row[k])
        cc = int(Jcoo.col[k])
        vv = -Jcoo.data[k]
        rows.append(row_block + rr)
        cols.append(col_block + cc)
        vals.append(vv)

    # right: (Beff ⊗ I)
    _add_A_kron_I(rows, cols, vals, Beff, r0, c_jp1, MK, scale=+1.0)

  if row_cursor != n:
    raise ValueError(f"row count mismatch row_cursor={row_cursor} n={n}")

  L = sp.coo_matrix(
    (np.array(vals, dtype=np.float64),
     (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
    shape=(n, n)
  ).tocsc()

  meta = {
    "D": D,
    "p": p,
    "K": K,
    "Mp": Mp,
    "MK": MK,
    "off_p": off_p,
    "blk_off": blk_off,
  }
  return L, meta

# --------------------------
# Lifted apply + test
# --------------------------

def apply_Mq_via_eq212(L, q_coeffs_p, c_coeffs_fun, Mp, Mfun):
  rhs = np.kron(q_coeffs_p, c_coeffs_fun)
  v = spla.spsolve(L.T, rhs)
  return v[:Mfun].copy()

def embed_coeffs_prefix(c_small, Mbig):
  out = np.zeros(Mbig, dtype=np.float64)
  out[:c_small.size] = c_small
  return out

def project_to_coeffs(D, n, kappa, nquad, fun):
  alpha, tail, invh, M = build_basis_structs(D, n, kappa)
  X, W = jquad_mapped_build_kappa(D, nquad, kappa)
  V = jbasis_eval_all(X, kappa, n, alpha, tail, invh, D)
  fx = fun(X)
  c = V.T @ (W * fx)
  return c, (X, W, V, alpha, tail, invh)

def test_eq212_lifted(D, N, p, kappa, nquad, f_fun, q_fun, K=None, tol=5e-12, verbose=True):
  if K is None:
    K = N + p  # dealias

  L, meta = build_L_otv_square_D(p, K, D, kappa)
  Mp = int(meta["Mp"])
  MK = int(meta["MK"])

  c_fN, (X, W, V_N, *_ ) = project_to_coeffs(D, N, kappa, nquad, f_fun)
  q_cp, _ = project_to_coeffs(D, p, kappa, nquad, q_fun)

  c_fK = embed_coeffs_prefix(c_fN, MK)

  y_K = apply_Mq_via_eq212(L, q_cp, c_fK, Mp, MK)
  y_N = y_K[:c_fN.size].copy()

  def qf_fun(Xin):
    return q_fun(Xin) * f_fun(Xin)

  c_qfN, _ = project_to_coeffs(D, N, kappa, nquad, qf_fun)

  rel_coef = np.linalg.norm(y_N - c_qfN) / max(1e-300, np.linalg.norm(c_qfN))
  qf_approx = V_N @ y_N
  qf_base = V_N @ c_qfN
  rel_L2 = np.sqrt(np.sum(W * (qf_approx - qf_base)**2) / max(1e-300, np.sum(W * qf_base**2)))

  if verbose:
    print(f"[eq212 lifted] D={D} N={N} p={p} K={K} nquad={nquad} rel_coef={rel_coef:.3e} rel_L2={rel_L2:.3e} "
          f"Lshape={L.shape} nnz={L.nnz}")

  ok = (rel_coef < tol) and (rel_L2 < tol)
  return ok, {"rel_coef": rel_coef, "rel_L2": rel_L2, "K": K, "L_nnz": L.nnz}


if __name__ == "__main__":
  D = 2
  kappa = np.array([0.8, 1.7, 2.3], dtype=np.float64)

  p = 10
  N = 5
  K = N + p
  nquad = K + p + 1  # your usual choice; bump if you want more headroom

  f_fun = lambda X: X[:,0]**5
  q_fun = lambda X: np.sin(X[:,0] + X[:,1])

  ok, info = test_eq212_lifted(D, N, p, kappa, nquad, f_fun, q_fun, K=K)

  D = 3
  kappa = np.array([0.8, 1.7, 2.3, 1.1], dtype=np.float64)
  p, N = 8, 4
  K = N + p
  nquad = K + p + 1
  
  f_fun = lambda X: X[:,0]**4
  q_fun = lambda X: np.sin(X[:,0] + X[:,1] + 0.5*X[:,2])
  
  ok, info = test_eq212_lifted(D, N, p, kappa, nquad, f_fun, q_fun)


