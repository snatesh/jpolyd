from poisson_tet import *

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
  """
  Merge two TetSteklovLeaf objects across a shared face Γ by enforcing:
    - trace continuity on Γ (single unknown λ_Γ in a chosen convention)
    - zero flux jump: μ_A + μ_B = 0 (outward normals)

  Convention:
    - We solve for λ_Γ in leaf A's face-moment convention.
    - Leaf B quantities are mapped into A's convention via P (orthonormal),
      where P maps moments from B -> A:
        λ_A = P λ_B,  μ_A = P μ_B.
  """

  def __init__(self,
               leafA,
               leafB,
               ifaceA,
               ifaceB,
               P_B_to_A=None,
               assume_P_orthonormal=True,
               check_P=True,
               check_tol=1e-12):
    self.leafA = leafA
    self.leafB = leafB
    self.ifaceA = int(ifaceA)
    self.ifaceB = int(ifaceB)

    if self.leafA.kf != self.leafB.kf:
      raise ValueError("leafA.kf != leafB.kf; cannot merge")
    self.kf = int(self.leafA.kf)

    # P maps B face moments -> A face moments.
    if P_B_to_A is None:
      # default: use precomputed table on ref (moments)
      sigmaA = self.leafA.face_sigma[self.ifaceA]
      sigmaB = self.leafB.face_sigma[self.ifaceB]
      # P_mom_ref is stored per face_id; use ifaceA (same polynomial space on any face)
      P_B_to_A = self.leafA.ref.P_mom_ref[self.ifaceA][sigmaA][sigmaB]
      #triA = LOCAL_FACE_TRIS[ifaceA]
      #triB = LOCAL_FACE_TRIS[ifaceB]
      
      #A_loc_glob = [leafA.gverts[i] for i in triA]
      #B_loc_glob = [leafB.gverts[i] for i in triB]
      
      #perm_Bloc_to_Aloc = tuple(A_loc_glob.index(g) for g in B_loc_glob)
      
      #sigma_B_to_A = tuple(perm_Bloc_to_Aloc[sigmaA[i]] for i in range(3))
      
      # Now use THIS to get P
      #P_B_to_A = leafA.ref.P_mom_ref[ifaceA][sigmaA][sigma_B_to_A]

    self.P = np.asarray(P_B_to_A, dtype=np.float64)
    if self.P.shape != (self.kf, self.kf):
      raise ValueError("P has wrong shape")
    print(self.P[np.abs(self.P)>1e-12])
    self.assume_P_orthonormal = bool(assume_P_orthonormal)

    if check_P:
      I = np.eye(self.kf)
      err = float(np.linalg.norm(self.P @ self.P.T - I))
      if err > check_tol:
        raise ValueError(f"P not orthonormal within tol: ||P P^T - I||={err:g}")

    # cache: interface system blocks (filled after build_interface_system)
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
    #self.P = np.eye(self.kf)

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


  def _P_inv(self):
    if self.assume_P_orthonormal:
      return self.P.T
    # fallback (not expected for your current P)
    return scipy.linalg.inv(self.P)

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
  
    Pinv = self._P_inv()
  
    KB_in_A = self.P @ KB @ Pinv
    hB_in_A = self.P @ hB
  
    K = KA + KB_in_A
    rhs = -(hA + hB_in_A)
  
    self.K = K
    self.rhs = rhs
    return K, rhs


  #def build_interface_system(self, lam_ext_A, f_int_A, lam_ext_B, f_int_B, Ctau=1.0, use_mass_from="A"):
  #  """
  #  Build the merged interface linear system for λ_Γ in A-convention.

  #  Inputs
  #  ------
  #  lam_ext_A : list length 4
  #    Dirichlet moments on leaf A faces (ifaceA entry may be None or ignored).
  #  f_int_A : (mA,)
  #    promoted RHS on leaf A.
  #  lam_ext_B : list length 4
  #    Dirichlet moments on leaf B faces (ifaceB entry may be None or ignored).
  #  f_int_B : (mB,)
  #    promoted RHS on leaf B.

  #  Produces
  #  --------
  #  K λ = rhs, where K = SA + P SB P^{-1} and rhs = -(gA + P gB).
  #  """
  #  ifaceA = self.ifaceA
  #  ifaceB = self.ifaceB

  #  SA, gA = self.leafA.build_dtn_face(ifaceA, lam_ext_blocks=lam_ext_A, f_int=f_int_A)
  #  SB, gB = self.leafB.build_dtn_face(ifaceB, lam_ext_blocks=lam_ext_B, f_int=f_int_B)

  #  P = self.P
  #  Pinv = self._P_inv()

  #  # Map B's DtN into A-convention
  #  # mu_B^A = P mu_B,  lam_B = Pinv lam_A
  #  # mu_B^A = (P SB Pinv) lam_A + (P gB)
  #  SB_in_A = P @ SB @ Pinv
  #  gB_in_A = P @ gB

  #  # Flux jump (outward normals): mu_A + mu_B^A = 0
  #  K = SA + SB_in_A
  #  rhs = -(gA + gB_in_A)
  #  tau = self.compute_tau(Ctau=Ctau)
  #  if tau != 0.0:
  #    tau = float(tau)
  #    if use_mass_from == "A":
  #      M = self.leafA.face_mass_matrix(self.ifaceA, sparse=False)
  #    elif use_mass_from == "B":
  #      MB = self.leafB.face_mass_matrix(self.ifaceB, sparse=False)
  #      M = self.P @ MB @ self.P.T
  #    else:
  #      raise ValueError("use_mass_from must be 'A' or 'B'")

  #    K = K + tau * M


    # store
    self.SA = SA
    self.gA = gA
    self.SB = SB
    self.gB = gB
    self.SB_in_A = SB_in_A
    self.gB_in_A = gB_in_A
    self.K = K
    self.rhs = rhs
    #info = analyze_K_spectrum(self.K, title=f"K ifaceA={self.ifaceA} ifaceB={self.ifaceB}")

    return K, rhs

  def solve_interface(self, use_sym_pos=False):
    """
    Solve for λ_Γ in A-convention.

    If you expect K to be SPD (often true for elliptic DtN with proper discretization),
    you can set use_sym_pos=True to use assume_a="sym" / SPD-ish paths. Here we just
    use a robust general solve.
    """
    if self.K is None or self.rhs is None:
      raise RuntimeError("Call build_interface_system first")

    K = self.K
    rhs = self.rhs

    # For now: robust dense solve
    if use_sym_pos:
      lam = scipy.linalg.solve(K, rhs, assume_a="sym", check_finite=False)
    else:

      #lam = scipy.linalg.solve(K, rhs, assume_a="gen", check_finite=False)
      lam, resid, rnk, s = scipy.linalg.lstsq(K, rhs, lapack_driver="gelsd", check_finite=False)
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
    Given λ_Γ in A-convention, compute volume coefficients (cA, cB).

    For leaf A: λ_ifaceA = lam_iface_A
    For leaf B: λ_ifaceB = P^{-1} lam_iface_A
    """
    if self.lam_iface_A is None:
      raise RuntimeError("Call solve_interface first")

    Pinv = self._P_inv()
    lamA = self.lam_iface_A
    lamB = Pinv @ lamA

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
      - trace continuity: λ_A - P λ_B
      - flux jump: μ_A + P μ_B
    """
    ifaceA = self.ifaceA
    ifaceB = self.ifaceB
    P = self.P

    lamA = np.asarray(self.leafA.trace_face(ifaceA, cA), dtype=np.float64).reshape(-1)
    lamB = np.asarray(self.leafB.trace_face(ifaceB, cB), dtype=np.float64).reshape(-1)
    muA = np.asarray(self.leafA.flux_face(ifaceA, cA), dtype=np.float64).reshape(-1)
    muB = np.asarray(self.leafB.flux_face(ifaceB, cB), dtype=np.float64).reshape(-1)

    tr_mis = lamA - (P @ lamB)
    fl_jmp = muA + (P @ muB)

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
  Two-tet convergence test. We build two affine tets sharing the face (v0,v1,v2):
    Tet A = (v0,v1,v2,v3)
    Tet B = (v0,v1,v2,v4)
  Interface face is local face_id=3 for both (opposite local vertex 3).
  """
  if kappa_src is None:
    kappa_src = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
  else:
    kappa_src = np.asarray(kappa_src, dtype=np.float64)

  u_exact, f_rhs, grad_u = make_manufactured_u_f_grad(m_max)

  # Shared face vertices
  v0 = np.array([0.20, -0.10, 0.30])
  v1 = np.array([1.10,  0.05, 0.20])
  v2 = np.array([0.10,  1.00, 0.40])

  # Two apex vertices on opposite sides (choose something reasonable)
  v3 = np.array([0.25,  0.20, 1.40])   # tet A apex
  v4 = np.array([0.10,  0.15, -0.80])  # tet B apex (roughly opposite side)

  VA = np.stack([v0, v1, v2, v3], axis=0)
  VB = np.stack([v4, v0, v1, v2], axis=0)

  # Global vertex IDs for adjacency bookkeeping
  tetA_g = (0, 1, 2, 3)
  tetB_g = (4, 0, 1, 2)

  sigA = build_face_sigma_list_for_tet(tetA_g)
  sigB = build_face_sigma_list_for_tet(tetB_g)

  shared_face_set = {0, 1, 2}
  ifaceA, gA = tet_local_face_and_ordered_gtriple(tetA_g, shared_face_set)
  ifaceB, gB = tet_local_face_and_ordered_gtriple(tetB_g, shared_face_set)
  print(gA)
  print(gB)
  q_err_min = int(m_max + q_pad)

  dofs = []
  errs = []

  for n in range(2, n_max + 1):
    q_vol = n + q_pad
    q_face = n + q_pad

    ref = RefTetPrecomp(n=n, q_vol=q_vol, q_face=q_face, kappa_src=kappa_src)
    leafA = TetSteklovLeaf(ref, VA, face_sigma=sigA, rtol_int=rtol_int, rtol_trace=rtol_trace)
    leafB = TetSteklovLeaf(ref, VB, face_sigma=sigB, rtol_int=rtol_int, rtol_trace=rtol_trace)
    leafA.gverts = tetA_g
    leafB.gverts = tetB_g
    XAf = leafA.ref.face[ifaceA]["Xf_hat_sigma"][leafA.face_sigma[ifaceA]]
    XBf = leafB.ref.face[ifaceB]["Xf_hat_sigma"][leafB.face_sigma[ifaceB]]
    
    # Map to physical
    XA_phys = leafA.map_hat_to_phys(XAf)   # or whatever your function name is
    XB_phys = leafB.map_hat_to_phys(XBf)
    
    print("max |XA_phys - XB_phys|:", np.max(np.linalg.norm(XA_phys - XB_phys, axis=1)))

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

    merge = MergeTwoLeaves(leafA, leafB, ifaceA, ifaceB, P_B_to_A=None, check_P=True)
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







