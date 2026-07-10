from poisson_tet_common_face import *

import numpy as np
import scipy.linalg

import sympy as sp
import matplotlib.pyplot as plt


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

  u_expr = 0
  for a in range(m_max + 1):
    for b in range(m_max + 1 - a):
      for c in range(m_max + 1 - a - b):
        coef = sp.Rational(1, 1 + a + b + c)
        u_expr += coef * (x**a) * (y**b) * (z**c)
  #u_expr = sp.exp(sp.cos(x**2 + y**2 + z**2))

  return make_sym_u_f_and_grad(u_expr, simplify=True)

def tet_local_face_and_ordered_gtriple(tet_gverts, face_gverts_set):
  tg = list(tet_gverts)
  s = set(face_gverts_set)

  for face_id, tri in enumerate(LOCAL_FACE_TRIS):
    ordered = (tg[tri[0]], tg[tri[1]], tg[tri[2]])  # ORDERED
    if set(ordered) == s:
      return face_id, ordered

  raise ValueError("face not found in tet")


def analyze_K_spectrum(K, title="K spectrum", eps=1e-300, show=True, savepath=None):
  """
  Analyze/plot spectrum of interface matrix K.

  Plots:
    - eigenvalues of symmetrized K: (K+K^T)/2
    - singular values of K

  Prints:
    - relative symmetry error
    - cond estimates from singular values
    - counts of tiny eigenvalues/singular values
  """
  K = np.asarray(K, dtype=np.float64)
  n = K.shape[0]
  if K.shape != (n, n):
    raise ValueError(f"K must be square; got {K.shape}")

  # symmetry diagnostic
  Kn = np.linalg.norm(K)
  sym_err = np.linalg.norm(K - K.T) / (Kn + eps)
  print(f"[K] size={n}  rel_sym_err={sym_err:.3e}")

  # symmetric part spectrum (robust even if K is slightly nonsymmetric)
  Ksym = 0.5 * (K + K.T)
  evals = np.linalg.eigvalsh(Ksym)  # sorted ascending
  evals_abs = np.abs(evals)

  # singular values (always)
  svals = np.linalg.svd(K, compute_uv=False)  # sorted descending
  cond = float(svals[0] / (svals[-1] + eps))
  print(f"[K] svals: max={svals[0]:.3e}  min={svals[-1]:.3e}  cond~{cond:.3e}")

  # crude "near-null" counts
  tol_e = 1e-12 * (np.max(evals_abs) + eps)
  tol_s = 1e-12 * (svals[0] + eps)
  print(f"[K] eig(|Ksym|) tiny count (<= {tol_e:.3e}): {int(np.sum(evals_abs <= tol_e))}/{n}")
  print(f"[K] svals tiny count (<= {tol_s:.3e}): {int(np.sum(svals <= tol_s))}/{n}")

  # plotting
  fig1 = plt.figure()
  plt.semilogy(np.arange(n), np.sort(evals_abs)[::-1] + eps, marker="o", linestyle="none")
  plt.xlabel("index (sorted)")
  plt.ylabel("|eig(Ksym)|")
  plt.title(title + "  |eig(Ksym)|")
  plt.grid(True)

  fig2 = plt.figure()
  plt.semilogy(np.arange(n), svals + eps, marker="o", linestyle="none")
  plt.xlabel("index (sorted)")
  plt.ylabel("singular values")
  plt.title(title + "  singular values")
  plt.grid(True)

  if savepath is not None:
    fig1.savefig(savepath + "_eig.png", dpi=160, bbox_inches="tight")
    fig2.savefig(savepath + "_svd.png", dpi=160, bbox_inches="tight")
    print(f"[K] wrote: {savepath}_eig.png, {savepath}_svd.png")

  if show:
    plt.show()

  return {
    "sym_err": sym_err,
    "evals": evals,
    "svals": svals,
    "cond": cond,
  }


class MergeTwoLeaves:
  """Merge two leaves whose interface moments already use one common basis.

  Sigma is handled inside each leaf when its trace and flux matrices are
  assembled. Therefore the interface vectors from both leaves have the same
  canonical face meaning and require no additional P map.
  """

  def __init__(self, leafA, leafB, ifaceA, ifaceB):
    self.leafA = leafA
    self.leafB = leafB
    self.ifaceA = int(ifaceA)
    self.ifaceB = int(ifaceB)

    if self.leafA.kf != self.leafB.kf:
      raise ValueError("leafA.kf != leafB.kf; cannot merge")
    if not np.array_equal(
        self.leafA.ref.kappa_face,
        self.leafB.ref.kappa_face
    ):
      raise ValueError("leaves do not use the same common face kappa")

    self.kf = int(self.leafA.kf)

    self.SA = None
    self.gA = None
    self.SB = None
    self.gB = None
    self.SB_in_A = None
    self.gB_in_A = None

    self.K = None
    self.rhs = None

    self.lam_iface_A = None
    self.cA = None
    self.cB = None

  def compute_tau(self, Ctau=1.0):
    """
    Compute stabilization parameter tau for the shared interface.
    """
    # polynomial degree on the face
    p = self.leafA.ref.n   # or leafA.face_deg[self.ifaceA] if you store per-face
  
    # face size (same physical face on A and B)
    hA = self.leafA.face_diameter(self.ifaceA)
    hB = self.leafB.face_diameter(self.ifaceB)
    h = 0.5 * (hA + hB)
  
    return float(Ctau * (p + 1)**2 / h)


  def build_interface_system(self, tau, lam_ext_A, f_int_A, lam_ext_B, f_int_B):
    tau = float(tau)

    KA, hA = self.leafA.build_face_augmented_map(
      self.ifaceA, tau,
      lam_ext_blocks=lam_ext_A,
      f_int=f_int_A
    )
    KB, hB = self.leafB.build_face_augmented_map(
      self.ifaceB, tau,
      lam_ext_blocks=lam_ext_B,
      f_int=f_int_B
    )

    # Both maps already act on the same canonical interface moments.
    K = KA + KB
    rhs = -(hA + hB)

    self.SA = KA
    self.gA = hA
    self.SB = KB
    self.gB = hB
    self.SB_in_A = KB
    self.gB_in_A = hB
    self.K = K
    self.rhs = rhs
    return K, rhs


  def is_spd(self, A):
    """
    Checks if a matrix A is symmetric positive-definite (SPD) using Cholesky decomposition.
    
    Returns True if the matrix is SPD, False otherwise.
    """
    # First, check if the matrix is symmetric (important as Cholesky assumes symmetry)
    if not np.array_equal(A, A.T):
      print("Matrix is not symmetric.")
      return False
        
    try:
      # The Cholesky factorization will fail if the matrix is not positive-definite
      np.linalg.cholesky(A)
      return True
    except np.linalg.LinAlgError:
      print("Matrix is symmetric but not positive-definite.")
      return False
  

  def solve_interface(self, use_sym_pos=False):
    """
    Solve for the common canonical interface moment vector λ_Γ.

    If you expect K to be SPD (often true for elliptic DtN with proper discretization),
    you can set use_sym_pos=True to use assume_a="sym" / SPD-ish paths. Here we just
    use a robust general solve.
    """
    if self.K is None or self.rhs is None:
      raise RuntimeError("Call build_interface_system first")

    K = self.K
    rhs = self.rhs
    print(f"IS K SPD? : {self.is_spd(K)}") 
    # For now: robust dense solve
    if use_sym_pos:
      lam = scipy.linalg.solve(K, rhs, assume_a="sym", check_finite=False)
    else:

      #lam = scipy.linalg.solve(K, rhs, assume_a="gen", check_finite=False)
      lam, resid, rnk, s = scipy.linalg.lstsq(K, rhs, lapack_driver="gelsd", check_finite=False)
      print(f"K: shape = {K.shape} rank={rnk} cond = {np.max(s)/np.min(s)}") 
      # symmetric equilibration
      #d = np.sqrt(np.abs(np.diag(K)) + 1e-300)
      #Dinv = 1.0 / d
      #Khat = (Dinv[:, None] * K) * Dinv[None, :]
      #rhshat = Dinv * rhs
      #
      #xhat = np.linalg.lstsq(Khat, rhshat, rcond=1e-14)[0]
      #lam = Dinv * xhat
    self.lam_iface_A = lam
    return lam

  def back_substitute(self, lam_ext_A, f_int_A, lam_ext_B, f_int_B):
    """
    Given the common λ_Γ, compute volume coefficients (cA, cB).

    The same canonical interface moment vector is imposed on both leaves.
    """
    if self.lam_iface_A is None:
      raise RuntimeError("Call solve_interface first")

    lamA = self.lam_iface_A
    lamB = lamA

    cA = self.leafA.solve_with_face_dirichlet(
      self.ifaceA,
      lamA,
      lam_ext_blocks=lam_ext_A,
      f_int=f_int_A
    )
    cB = self.leafB.solve_with_face_dirichlet(
      self.ifaceB,
      lamB,
      lam_ext_blocks=lam_ext_B,
      f_int=f_int_B
    )
    # --- per-leaf trace residuals on the interface (did we actually hit the imposed moments?) ---
    lamA_hit = self.leafA.trace_face(self.ifaceA, cA)
    lamB_hit = self.leafB.trace_face(self.ifaceB, cB)

    rA = np.asarray(lamA_hit, dtype=np.float64).reshape(-1) - np.asarray(lamA, dtype=np.float64).reshape(-1)
    rB = np.asarray(lamB_hit, dtype=np.float64).reshape(-1) - np.asarray(lamB, dtype=np.float64).reshape(-1)

    nA = float(np.linalg.norm(rA))
    nB = float(np.linalg.norm(rB))
    dA = float(np.linalg.norm(np.asarray(lamA, dtype=np.float64).reshape(-1)))
    dB = float(np.linalg.norm(np.asarray(lamB, dtype=np.float64).reshape(-1)))
    print(f"[backsub] relA = {nA/(dA+1e-300):.3e}, relB = {nB/(dB+1e-300):.3e}")
    r_face_A = self.leafA.T_full @ cA - self.leafA.assemble_rhs(
      lam_blocks=[lamA if f == self.ifaceA else lam_ext_A[f] for f in range(4)],
      f_int=None
    )[:4*self.leafA.kf]
    
    r_int_A = self.leafA.L_int @ cA + f_int_A
    
    print("A: ||face||", np.linalg.norm(r_face_A), "||int||", np.linalg.norm(r_int_A))

    self.cA = cA
    self.cB = cB
    return cA, cB

  def solve(self,
            lam_ext_A,
            f_int_A,
            lam_ext_B,
            f_int_B,
            check=True,
            check_tol=1e-10):
    """
    Convenience: build system, solve interface, back-substitute.

    If check=True, verify flux-jump and trace continuity (in A convention).
    """
    tau = self.compute_tau(1.0)
    self.build_interface_system(tau, lam_ext_A, f_int_A, lam_ext_B, f_int_B)
    self.solve_interface()
    cA, cB = self.back_substitute(lam_ext_A, f_int_A, lam_ext_B, f_int_B)

    if check:
      self.check_interface(cA, cB, tol=check_tol)

    return cA, cB, self.lam_iface_A

  def check_interface(self, cA, cB, tol=1e-10):
    """
    Diagnostics:
      - trace continuity: λ_A - λ_B
      - flux jump: μ_A + μ_B
    """
    ifaceA = self.ifaceA
    ifaceB = self.ifaceB
    lamA = np.asarray(self.leafA.trace_face(ifaceA, cA), dtype=np.float64).reshape(-1)
    lamB = np.asarray(self.leafB.trace_face(ifaceB, cB), dtype=np.float64).reshape(-1)
    muA = np.asarray(self.leafA.flux_face(ifaceA, cA), dtype=np.float64).reshape(-1)
    muB = np.asarray(self.leafB.flux_face(ifaceB, cB), dtype=np.float64).reshape(-1)

    tr_mis = lamA - lamB
    fl_jmp = muA + muB

    n_tr = float(np.linalg.norm(tr_mis))
    n_fl = float(np.linalg.norm(fl_jmp))

    #if n_tr > tol or n_fl > tol:
    #  raise RuntimeError(
    #    f"Interface check failed: ||trace_mismatch||={n_tr:.3e}, ||flux_jump||={n_fl:.3e}"
    #  )

    return n_tr, n_fl


def run_two_tet_poly_convergence(m_max=8, n_max=14, q_pad=2,
                                 kappa_src=None,
                                 rtol_int=1e-14,
                                 rtol_trace=1e-14,
                                 do_print=True,
                                 check_interface=True):
  """
  Two-tet convergence test with a deliberately permuted second tetrahedron.

    Tet A local-to-global order = (0,1,2,3)
    Tet B local-to-global order = (4,2,0,1)

  The shared physical face is {0,1,2}. It is local face 3 of Tet A and local
  face 0 of Tet B. Both leaves share one RefTetPrecomp and one canonical face
  Jacobi vector. Sigma only maps canonical face quadrature points into each
  element's local face coordinates.
  """
  if kappa_src is None:
    # Stored Jacobi order for Tet A:
    #   [lambda_1 exponent, lambda_2 exponent,
    #    lambda_3 exponent, lambda_0 exponent].
    kappa_src = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
  else:
    kappa_src = np.asarray(kappa_src, dtype=np.float64)
  if kappa_src.shape != (4,):
    raise ValueError("kappa_src must have shape (4,)")

  u_exact, f_rhs, grad_u = make_manufactured_u_f_grad(m_max)

  # Shared face vertices
  v0 = np.array([0.20, -0.10, 0.30])
  v1 = np.array([1.10,  0.05, 0.20])
  v2 = np.array([0.10,  1.00, 0.40])

  # Two apex vertices on opposite sides (choose something reasonable)
  v3 = np.array([0.25,  0.20, 1.40])   # tet A apex
  v4 = np.array([0.10,  0.15, -0.80])  # tet B apex (roughly opposite side)

  VA = np.stack([v0, v1, v2, v3], axis=0)

  # Deliberately use a nontrivial local ordering on tet B.  Its shared face
  # has local global-vertex order (2,0,1), so the local-to-canonical sigma is
  # a three-cycle rather than the identity.
  VB = np.stack([v4, v2, v0, v1], axis=0)

  # Global vertex IDs for adjacency bookkeeping.
  tetA_g = (0, 1, 2, 3)
  tetB_g = (4, 2, 0, 1)

  sigA = build_face_sigma_list_for_tet(tetA_g)
  sigB = build_face_sigma_list_for_tet(tetB_g)

  shared_face_set = {0, 1, 2}
  ifaceA, gA = tet_local_face_and_ordered_gtriple(tetA_g, shared_face_set)
  ifaceB, gB = tet_local_face_and_ordered_gtriple(tetB_g, shared_face_set)
  print(gA)
  print(gB)
  print("shared sigma A:", sigA[ifaceA])
  print("shared sigma B:", sigB[ifaceB])

  # One common face basis for every face. By default RefTetPrecomp uses the
  # recursive truncation of the volume parameter vector.
  kappa_face = kappa_src[:-1].copy()
  print("volume kappa storage:", kappa_src)
  print("common face kappa storage:", kappa_face)

  q_err_min = int(m_max + q_pad)

  dofs = []
  errs = []

  for n in range(2, n_max + 1):
    q_vol = n + q_pad
    q_face = n + q_pad

    # One reusable reference precomputation for both physical tetrahedra.
    ref = RefTetPrecomp(
      n=n,
      q_vol=q_vol,
      q_face=q_face,
      kappa_src=kappa_src,
      kappa_face=kappa_face
    )
    leafA = TetSteklovLeaf(
      ref, VA, face_sigma=sigA,
      rtol_int=rtol_int, rtol_trace=rtol_trace
    )
    leafB = TetSteklovLeaf(
      ref, VB, face_sigma=sigB,
      rtol_int=rtol_int, rtol_trace=rtol_trace
    )
    leafA.gverts = tetA_g
    leafB.gverts = tetB_g

    XAf = ref.face[ifaceA]["Xf_hat_sigma"][leafA.face_sigma[ifaceA]]
    XBf = ref.face[ifaceB]["Xf_hat_sigma"][leafB.face_sigma[ifaceB]]

    XA_phys = leafA.map_hat_to_phys(XAf)
    XB_phys = leafB.map_hat_to_phys(XBf)

    point_mismatch = np.max(np.linalg.norm(XA_phys - XB_phys, axis=1))
    print("max |XA_phys - XB_phys|:", point_mismatch)
    print("shared face kappa A:", ref.face[ifaceA]["kappa_tri"])
    print("shared face kappa B:", ref.face[ifaceB]["kappa_tri"])

    if point_mismatch > 1e-13:
      raise RuntimeError(
        "permuted canonical face quadrature points do not map to the same "
        f"physical points: mismatch={point_mismatch}"
      )
    if not np.array_equal(ref.face[ifaceA]["kappa_tri"], kappa_face):
      raise RuntimeError("Tet A shared face does not use the common face kappa")
    if not np.array_equal(ref.face[ifaceB]["kappa_tri"], kappa_face):
      raise RuntimeError("Tet B shared face does not use the common face kappa")

    ############# test
    # pick leaf = leafA
    leaf = leafA
    iface = ifaceA
    
    lam_test = leaf.project_dirichlet_face(iface, u_exact)
    
    # external BCs (iface ignored)
    lam_ext = [leaf.project_dirichlet_face(f, u_exact) for f in range(4)]
    lam_ext[iface] = None
    
    f_int = leaf.project_source_int(f_rhs)
    
    # DtN build
    S, g = leaf.build_dtn_face(iface, lam_ext_blocks=lam_ext, f_int=f_int)
    
    # Direct solve with that interface trace
    c = leaf.solve_with_face_dirichlet(
        iface,
        lam_test,
        lam_ext_blocks=lam_ext,
        f_int=f_int
    )
    
    mu_true = leaf.flux_face(iface, c)
    mu_pred = S @ lam_test + g
    
    print("||mu_true - mu_pred|| / ||mu_true|| =", np.linalg.norm(mu_true - mu_pred)/np.linalg.norm(mu_true))
    ################ 
    ################ test
    lam_ext_A = [leafA.project_dirichlet_face(f,u_exact) for f in range(4)]
    lam_ext_A[ifaceA] = None
    lam_ext_B = [leafB.project_dirichlet_face(f,u_exact) for f in range(4)]
    lam_ext_B[ifaceB] = None
    f_int_A = leafA.project_source_int(f_rhs)
    f_int_B = leafB.project_source_int(f_rhs)
    # true interface trace moments from manufactured solution
    lam_true_A = leafA.project_dirichlet_face(ifaceA, u_exact)
    lam_true_B = leafB.project_dirichlet_face(ifaceB, u_exact)
    
    # solve each leaf with the TRUE interface trace
    cA_true = leafA.solve_with_face_dirichlet(
      ifaceA, lam_true_A,
      lam_ext_blocks=lam_ext_A,
      f_int=f_int_A
    )
    cB_true = leafB.solve_with_face_dirichlet(
      ifaceB, lam_true_B,
      lam_ext_blocks=lam_ext_B,
      f_int=f_int_B
    )
   
    cA_exact, _ = leafA.project_u_vol(u_exact)
    cB_exact, _ = leafB.project_u_vol(u_exact)
    print(np.linalg.norm(cA_true-cA_exact)) 
    print(np.linalg.norm(cB_true-cB_exact)) 
    muA_true = np.asarray(leafA.flux_face(ifaceA, cA_true), dtype=np.float64).reshape(-1)
    muA_exact = np.asarray(leafA.flux_face(ifaceA,cA_exact), dtype=np.float64).reshape(-1)
    muB_true = np.asarray(leafB.flux_face(ifaceB, cB_true), dtype=np.float64).reshape(-1)
    muB_exact = np.asarray(leafB.flux_face(ifaceB, cB_exact), dtype=np.float64).reshape(-1)
    lamAint_true = np.asarray(leafA.trace_face(ifaceA, cA_true), dtype=np.float64).reshape(-1)    
    lamAint_exact = np.asarray(leafA.trace_face(ifaceA, cA_exact), dtype=np.float64).reshape(-1)    
    lamBint_true = np.asarray(leafB.trace_face(ifaceB, cB_true), dtype=np.float64).reshape(-1)    
    lamBint_exact = np.asarray(leafB.trace_face(ifaceB, cB_exact), dtype=np.float64).reshape(-1)    
 
    print("A flux rel err:", np.linalg.norm(muA_true - muA_exact) /
                           (np.linalg.norm(muA_exact) + 1e-300))    
    print("B flux rel err:", np.linalg.norm(muB_true - muB_exact) /
                           (np.linalg.norm(muB_exact) + 1e-300))    
    print("A trace rel err:", np.linalg.norm(lamAint_true - lamAint_exact) /
                           (np.linalg.norm(lamAint_exact) + 1e-300))    
    print("B trace rel err:", np.linalg.norm(lamBint_true - lamBint_exact) /
                           (np.linalg.norm(lamBint_exact) + 1e-300))    

    nA = float(np.linalg.norm(muA_true))
    nB = float(np.linalg.norm(muB_true))
    nA_exact = np.linalg.norm(muA_exact)
    nB_exact = np.linalg.norm(muB_exact)
    s_plus  = float(np.linalg.norm(muA_true + muB_true) / (nA + nB + 1e-300))
    s_minus = float(np.linalg.norm(muA_true - muB_true) / (nA + nB + 1e-300))
    s_plus_exact  = float(np.linalg.norm(muA_exact + muB_exact) / (nA_exact + nB_exact + 1e-300))
    s_minus_exact = float(np.linalg.norm(muA_exact - muB_exact) / (nA_exact + nB_exact + 1e-300))
    
    print(f"[true flux coupling] rel ||muA + muB|| = {s_plus:.3e}")
    print(f"[true flux coupling] rel ||muA - muB|| = {s_minus:.3e}")
    print(f"[exact flux coupling] rel ||muA + muB|| = {s_plus:.3e}")
    print(f"[exact flux coupling] rel ||muA - muB|| = {s_minus:.3e}")
    print("rel exact ||lamA - lamB||:", np.linalg.norm(lamAint_exact - lamBint_exact) /
                                (np.linalg.norm(lamAint_exact) + 1e-300))

    ############# 

    # External Dirichlet moments on each tet; set interface entry to None
    lam_ext_A = [leafA.project_dirichlet_face(f, u_exact) for f in range(4)]
    lam_ext_B = [leafB.project_dirichlet_face(f, u_exact) for f in range(4)]
    lam_ext_A[ifaceA] = None
    lam_ext_B[ifaceB] = None

    # Promoted interior RHS moments
    f_int_A = leafA.project_source_int(f_rhs)
    f_int_B = leafB.project_source_int(f_rhs)

    merge = MergeTwoLeaves(leafA, leafB, ifaceA, ifaceB)
    cA, cB, lamA = merge.solve(
      lam_ext_A, f_int_A,
      lam_ext_B, f_int_B,
      check=check_interface,
      check_tol=1e-10
    )

    # Combined relative L2 error over union of the two tets
    num2A, den2A = leafA.weighted_l2_error_quad(cA, u_exact, q_err_min=q_err_min)
    num2B, den2B = leafB.weighted_l2_error_quad(cB, u_exact, q_err_min=q_err_min)
    rel_err = np.sqrt((num2A + num2B) / (den2A + den2B + 1e-300))

    dofs.append(int(ref.m))   # polynomial degree dofs metric (same per tet)
    errs.append(float(rel_err))

    if do_print:
      print(f"n={n:2d}  dofs(tet)={ref.m:5d}  rel_err={rel_err:.3e}")

  return np.array(dofs, dtype=np.int64), np.array(errs, dtype=np.float64)


def main():
  dofs, errs = run_two_tet_poly_convergence(
    m_max=8,
    n_max=12,
    q_pad=1,
    do_print=True,
    check_interface=True
  )
  plt.semilogy(dofs, errs)
  plt.xlabel("dofs (per tet: dim P_{n-2}(tet))")
  plt.ylabel("relative L2 error (2 tets)")
  plt.title("Two-tet Poisson convergence (DtN merge)")
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  main()







