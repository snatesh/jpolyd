import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from jmat import *
from jbasis import *
from jquad_tprod import *

# --------------------------------------------------------------------
# Helpers imported / mirrored from construct_LD_general.py
# (kept local so this file is standalone in your testing folder)
# --------------------------------------------------------------------

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
  return comb_int(j + D - 1, D - 1)

def pivot_rows_D_canonical(alpha_table_p, off_p, j, D):
  s0 = int(off_p[j])
  s1 = int(off_p[j + 1])
  mj = s1 - s0
  alpha_j = alpha_table_p[s0:s1, :]  # (mj, D)

  piv = []
  piv.extend(list(range(0, mj)))
  for i in range(1, D):
    prefix = alpha_j[:, :i]
    sel = np.where(np.all(prefix == 0, axis=1))[0].astype(np.int64)
    piv.extend((i * mj + sel).tolist())

  piv = np.array(piv, dtype=np.int64)
  mj1 = m_hom(D, j + 1)
  if piv.size != mj1:
    raise ValueError(f"pivot size mismatch at j={j}: got {piv.size}, expected {mj1}")
  return piv

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

# --------------------------------------------------------------------
# Sparse triangular solve (upper triangular CSR) with many RHS
# --------------------------------------------------------------------

def solve_upper_tri_sparse_many_rhs(Bt_csr, RHS, check_diag=True):
  """
  Solve Bt * X = RHS for X, where:
    - Bt_csr is sparse CSR, (n,n), assumed upper triangular (cols >= row)
    - RHS is dense (n, nrhs)

  Returns X dense (n, nrhs).

  Complexity: O(nnz(Bt) * nrhs). With nnz/row = O(1), this is ~O(n * nrhs).
  """
  Bt = Bt_csr
  if not sp.isspmatrix_csr(Bt):
    Bt = Bt.tocsr()

  n = Bt.shape[0]
  if Bt.shape[1] != n:
    raise ValueError("Bt must be square")
  if RHS.shape[0] != n:
    raise ValueError("RHS row dimension mismatch")

  X = RHS.copy()

  indptr = Bt.indptr
  indices = Bt.indices
  data = Bt.data

  # Back substitution: i = n-1..0
  for i in range(n - 1, -1, -1):
    rs = indptr[i]
    re = indptr[i + 1]

    diag = 0.0
    have_diag = False

    # Row i: sum_{j>i} Bt[i,j] X[j,:] + Bt[i,i] X[i,:] = RHS[i,:]
    # We'll subtract the j>i terms from X[i,:] (currently RHS), then divide by diag.
    for k in range(rs, re):
      j = int(indices[k])
      a = float(data[k])
      if j == i:
        diag = a
        have_diag = True
      elif j > i:
        X[i, :] -= a * X[j, :]
      else:
        # If numerical garbage appears below diagonal, treat as error.
        # You can relax this if needed.
        X[i, :] -= a * X[j, :]

    if check_diag and (not have_diag):
      raise RuntimeError(f"missing diagonal at row {i} in Bt")
    if diag == 0.0:
      raise RuntimeError(f"zero diagonal at row {i} in Bt")

    X[i, :] /= diag

  return X

# --------------------------------------------------------------------
# Block back-substitution (operator Clenshaw) for Eq. (OTV 2.12 analogue)
# --------------------------------------------------------------------

class MultByQClenshaw:
  """
  Implements the block back-substitution for the transposed triangular system

    L_{p,K}(J)^T v = q ⊗ c^(K)

  without assembling L_{p,K}(J). The output is v0 = M_q^(K) c^(K).

  Matches construct_LD_general.py exactly:
    - canonical pivot rule
    - Aeff/Beff/Ceff extracted from J_poly (degree p)
    - row-picked -(I ⊗ J_fun_i) operator using J_fun (degree K)
  """

  def __init__(self, D, p, K, kappa, assume_symmetric=True):
    self.D = int(D)
    self.p = int(p)
    self.K = int(K)
    self.kappa = np.asarray(kappa, dtype=np.float64)
    self.assume_symmetric = bool(assume_symmetric)

    # poly basis ordering for q
    alpha_p, tail_p, invh_p, Mp = build_basis_structs(self.D, self.p, self.kappa)
    off_p = degree_offsets_from_alpha(alpha_p, self.p)

    # function space size
    alpha_K, tail_K, invh_K, MK = build_basis_structs(self.D, self.K, self.kappa)

    self.alpha_p = alpha_p
    self.off_p = off_p
    self.Mp = int(Mp)
    self.MK = int(MK)

    # Jacobi matrices
    self.J_poly = [A.tocsc() for A in jmat_build_csc(self.D, self.p, self.kappa)]
    self.J_fun  = [A.tocsc() for A in jmat_build_csc(self.D, self.K, self.kappa)]

    # Per-level effective blocks and sigma maps (j=0..p-1)
    self.Aeff = [None] * self.p     # (m_{j+1} x m_j)
    self.Beff = [None] * self.p     # (m_{j+1} x m_{j+1})
    self.Ceff = [None] * self.p     # (m_{j+1} x m_{j-1}) for j>=1 else None

    self.sigma_i = [None] * self.p  # coord index per selected row, length m_{j+1}
    self.sigma_b = [None] * self.p  # within-degree index per selected row, length m_{j+1}

    # Sparse CSR caches for Bt = Beff^T (upper triangular)
    # Used to solve (Beff^T ⊗ I) v = s at each backsub step.
    self.BT_csr = [None] * self.p   # Bt for level j: Beff[j].T (size m_{j+1} x m_{j+1})

    for j in range(self.p):
      self._precompute_level(j)

  def _precompute_level(self, j):
    D = self.D
    off_p = self.off_p
    alpha_p = self.alpha_p

    mj = m_hom(D, j)
    mj1 = m_hom(D, j + 1)

    sj0, sj1 = int(off_p[j]), int(off_p[j + 1])
    sjp10, sjp11 = int(off_p[j + 1]), int(off_p[j + 2])

    A_blocks = []
    B_blocks = []
    C_blocks = [] if j >= 1 else None

    for i in range(D):
      Ji_p = self.J_poly[i]
      A_blocks.append(Ji_p[sj0:sj1, sj0:sj1].tocsc())
      B_blocks.append(Ji_p[sj0:sj1, sjp10:sjp11].tocsc())

      if j >= 1:
        sm10, sm11 = int(off_p[j - 1]), int(off_p[j])
        if self.assume_symmetric:
          Bj_m1 = Ji_p[sm10:sm11, sj0:sj1].tocsc()
          C_blocks.append(Bj_m1.T.tocsc())
        else:
          C_blocks.append(Ji_p[sj0:sj1, sm10:sm11].tocsc())

    Astack = sp.vstack(A_blocks, format="csc")   # (D*mj) x mj
    Bstack = sp.vstack(B_blocks, format="csc")   # (D*mj) x mj1
    Cstack = sp.vstack(C_blocks, format="csc") if j >= 1 else None  # (D*mj) x m_{j-1}

    piv = pivot_rows_D_canonical(alpha_p, off_p, j, D)  # length mj1

    Aeff = Astack[piv, :].tocsc()
    Beff = Bstack[piv, :].tocsc()
    Ceff = Cstack[piv, :].tocsc() if j >= 1 else None

    self.Aeff[j] = Aeff
    self.Beff[j] = Beff
    self.Ceff[j] = Ceff

    # sigma mapping (row-picked -(I ⊗ J_fun_i))
    sigma_i = np.empty(mj1, dtype=np.int64)
    sigma_b = np.empty(mj1, dtype=np.int64)
    for r_sel in range(mj1):
      r_stack = int(piv[r_sel])
      i = r_stack // mj
      b = r_stack - i * mj
      sigma_i[r_sel] = i
      sigma_b[r_sel] = b

    self.sigma_i[j] = sigma_i
    self.sigma_b[j] = sigma_b

    # Cache Bt = Beff^T as CSR for fast upper-triangular back-sub
    Bt = Beff.T.tocsr()
    Bt.sort_indices()
    self.BT_csr[j] = Bt

  # --------------------------
  # Operator applications
  # --------------------------

  def _apply_AkronI_T(self, j, w):
    """
    Compute (Aeff[j] ⊗ I)^T w, where w in R^{m_{j+1}*MK}, output in R^{m_j*MK}.
    """
    MK = self.MK
    A = self.Aeff[j].toarray()          # (m_{j+1}, m_j), small dense
    mj = A.shape[1]
    W = w.reshape((-1, MK), order="C")  # (m_{j+1}, MK)
    Y = A.T @ W                         # (m_j, MK)
    return Y.reshape((mj * MK,), order="C")

  def _apply_CkronI_T(self, j_plus_1, w):
    """
    Compute (Ceff[j_plus_1] ⊗ I)^T w.
    Here Ceff[j_plus_1] has shape (m_{j+2}, m_j).
    w is v_{j+2} in R^{m_{j+2}*MK}; output is in R^{m_j*MK}.
    """
    MK = self.MK
    C = self.Ceff[j_plus_1]
    if C is None:
      raise ValueError("Ceff is None for this level")
    Cd = C.toarray()                    # (m_{j+2}, m_j), small dense
    mj = Cd.shape[1]
    W = w.reshape((-1, MK), order="C")  # (m_{j+2}, MK)
    Y = Cd.T @ W                        # (m_j, MK)
    return Y.reshape((mj * MK,), order="C")

  def _apply_JT(self, j, w):
    """
    Compute J_j^T w using the canonical row map (sigma).
    J_j maps R^{m_j*MK} -> R^{m_{j+1}*MK} by row-picked J_fun blocks.
    We apply the adjoint: R^{m_{j+1}*MK} -> R^{m_j*MK}.

    w is in R^{m_{j+1}*MK}. Output is in R^{m_j*MK}.
    """
    MK = self.MK
    sigma_i = self.sigma_i[j]
    sigma_b = self.sigma_b[j]

    mj = m_hom(self.D, j)
    mj1 = w.size // MK
    if mj1 != sigma_i.size:
      raise ValueError("w size does not match sigma size")

    out = np.zeros((mj, MK), dtype=np.float64)
    W = w.reshape((mj1, MK), order="C")

    for ell in range(mj1):
      i = int(sigma_i[ell])
      b = int(sigma_b[ell])
      out[b, :] += self.J_fun[i].dot(W[ell, :])

    return out.reshape((mj * MK,), order="C")

  def _solve_BT_kronI(self, j_minus_1, s):
    """
    Solve (Beff[j_minus_1]^T ⊗ I) v = s using sparse upper-triangular back-sub.

    Here Bt = Beff[j_minus_1].T is size (m_j x m_j) with j = j_minus_1+1.
    s is length m_j*MK. Output v same length.
    """
    MK = self.MK
    mj = s.size // MK
    S = s.reshape((mj, MK), order="C")

    Bt = self.BT_csr[j_minus_1]
    if Bt.shape[0] != mj:
      raise ValueError("Bt shape mismatch for this solve")

    V = solve_upper_tri_sparse_many_rhs(Bt, S)
    return V.reshape((mj * MK,), order="C")

  # --------------------------
  # Main apply: v0 = M_q^(K) c^(K)
  # --------------------------

  def apply(self, q_coeffs_p, c_coeffs_K):
    """
    Compute y_K = M_q^(K) c_coeffs_K as v0 from the back substitution.

    Inputs:
      q_coeffs_p : (Mp,) coefficients of q in Pi_p
      c_coeffs_K : (MK,) coefficients of f embedded in Pi_K

    Returns:
      y_K : (MK,) coefficients of Pi_K projection of q*f (de-aliased if K>=N+p)
    """
    q_coeffs_p = np.asarray(q_coeffs_p, dtype=np.float64)
    c_coeffs_K = np.asarray(c_coeffs_K, dtype=np.float64)
    if q_coeffs_p.size != self.Mp:
      raise ValueError("q_coeffs_p size mismatch")
    if c_coeffs_K.size != self.MK:
      raise ValueError("c_coeffs_K size mismatch")

    # RHS blocks r_j = q_j ⊗ c_K
    r = [None] * (self.p + 1)
    for j in range(self.p + 1):
      s0 = int(self.off_p[j])
      s1 = int(self.off_p[j + 1])
      qj = q_coeffs_p[s0:s1]
      r[j] = np.kron(qj, c_coeffs_K)

    v = [None] * (self.p + 3)
    v[self.p + 1] = np.zeros(0, dtype=np.float64)
    v[self.p + 2] = np.zeros(0, dtype=np.float64)

    if self.p == 0:
      return r[0].copy()

    # v_p: (Beff[p-1]^T ⊗ I) v_p = r_p
    v[self.p] = self._solve_BT_kronI(self.p - 1, r[self.p])

    # v_{p-1}..v_1
    for j in range(self.p - 1, 0, -1):
      term_A = self._apply_AkronI_T(j, v[j + 1])
      term_J = self._apply_JT(j, v[j + 1])
      s = r[j] - (term_A - term_J)

      if j + 1 <= self.p - 1:
        if (j + 1) >= 1:
          s -= self._apply_CkronI_T(j + 1, v[j + 2])

      v[j] = self._solve_BT_kronI(j - 1, s)

    # v0
    term_A0 = self._apply_AkronI_T(0, v[1])
    term_J0 = self._apply_JT(0, v[1])
    v0 = r[0] - (term_A0 - term_J0)

    if self.p >= 2:
      v0 -= self._apply_CkronI_T(1, v[2])

    return v0.copy()

# --------------------------------------------------------------------
# Convenience wrappers: build + apply + end-to-end test
# --------------------------------------------------------------------

def multbyq_apply_clenshaw(D, N, p, kappa, c_fN, q_cp, K=None):
  if K is None:
    K = N + p
  solver = MultByQClenshaw(D, p, K, kappa)
  c_fK = embed_coeffs_prefix(c_fN, solver.MK)
  y_K = solver.apply(q_cp, c_fK)
  return y_K[:c_fN.size].copy()

def test_eq212_lifted_backsub(D, N, p, kappa, nquad, f_fun, q_fun, K=None, tol=5e-12, verbose=True):
  if K is None:
    K = N + p

  c_fN, (X, W, V_N, *_ ) = project_to_coeffs(D, N, kappa, nquad, f_fun)
  q_cp, _ = project_to_coeffs(D, p, kappa, nquad, q_fun)

  y_N = multbyq_apply_clenshaw(D, N, p, kappa, c_fN, q_cp, K=K)

  def qf_fun(Xin):
    return q_fun(Xin) * f_fun(Xin)

  c_qfN, _ = project_to_coeffs(D, N, kappa, nquad, qf_fun)

  rel_coef = np.linalg.norm(y_N - c_qfN) / max(1e-300, np.linalg.norm(c_qfN))
  qf_approx = V_N @ y_N
  qf_base = V_N @ c_qfN
  rel_L2 = np.sqrt(np.sum(W * (qf_approx - qf_base)**2) / max(1e-300, np.sum(W * (qf_base**2))))

  if verbose:
    print(f"[eq212 backsub] D={D} N={N} p={p} K={K} nquad={nquad} rel_coef={rel_coef:.3e} rel_L2={rel_L2:.3e}")

  ok = (rel_coef < tol) and (rel_L2 < tol)
  return ok, {"rel_coef": rel_coef, "rel_L2": rel_L2}

import time

def scaling_study_D3():
  D = 3
  kappa = np.array([0.8, 1.7, 2.3, 1.1], dtype=np.float64)

  # Choose a modest set first; MK grows ~ O(K^D) so this ramps fast.
  Ns = [4, 6, 8, 10, 12]

  print("\n[D=3 scaling] timing construction + one apply")
  print("  cols: N p K  Mp MK  t_build(s) t_apply(s)  nnz/row(Bt) approx")
  print("  ------------------------------------------------------------")

  for N in Ns:
    p = N  # try p=N (harder); or p=max(1, N-1)
    K = N + p

    # Build solver (this includes building J_poly/J_fun + precompute_level over j)
    t0 = time.perf_counter()
    solver = MultByQClenshaw(D, p, K, kappa)
    t1 = time.perf_counter()
    t_build = t1 - t0

    Mp = solver.Mp
    MK = solver.MK

    # Construct a cheap test (no quadrature needed for timing apply):
    # Use random coefficients (or structured) to avoid being dominated by sin eval etc.
    rng = np.random.default_rng(0)
    q_cp = rng.standard_normal(Mp).astype(np.float64)
    c_fK = rng.standard_normal(MK).astype(np.float64)

    # Warmup apply (optional, helps stabilize cache effects)
    _ = solver.apply(q_cp, c_fK)

    # Time one apply
    t2 = time.perf_counter()
    _ = solver.apply(q_cp, c_fK)
    t3 = time.perf_counter()
    t_apply = t3 - t2

    # quick nnz/row estimate from last Bt
    # pick a representative level, say mid-level jmid where Bt corresponds to Beff[jmid].T
    jmid = max(0, min(p - 1, p // 2))
    Bt = solver.BT_csr[jmid]
    nnz_per_row = Bt.nnz / max(1, Bt.shape[0])

    print(f"  {N:2d} {p:2d} {K:2d}  {Mp:6d} {MK:7d}  {t_build:9.3e} {t_apply:9.3e}   {nnz_per_row:5.2f}")


if __name__ == "__main__":
  
  D = 1
  kappa = np.array([0.8, 1.7], dtype=np.float64)

  p = 10
  N = 5
  K = N + p
  nquad = K + p + 1  # your usual choice; bump if you want more headroom

  f_fun = lambda X: X[:,0]**5
  q_fun = lambda X: np.sin(X[:,0])

  ok, info = test_eq212_lifted_backsub(D, N, p, kappa, nquad, f_fun, q_fun, K=K)
  
  D = 2
  kappa = np.array([0.8, 1.7, 2.3], dtype=np.float64)

  p = 10
  N = 5
  K = N + p
  nquad = K + p + 1

  f_fun = lambda X: X[:,0]**5
  q_fun = lambda X: np.sin(X[:,0] + X[:,1])

  ok, info = test_eq212_lifted_backsub(D, N, p, kappa, nquad, f_fun, q_fun, K=K)

  # D=3 smoke
  D = 3
  kappa = np.array([0.8, 1.7, 2.3, 1.1], dtype=np.float64)

  p, N = 10, 8
  K = N + p
  nquad = K + p + 1

  f_fun = lambda X: X[:,0]**4
  q_fun = lambda X: np.sin(X[:,0] + X[:,1] + 0.5*X[:,2])

  ok, info = test_eq212_lifted_backsub(D, N, p, kappa, nquad, f_fun, q_fun, K=K)
  
  scaling_study_D3()



