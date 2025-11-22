import numpy as np
from jmat import jmat_build
from jevd import jevd_serial
from jbasis import (
  jbasis_build_structures, 
  jbasis_eval_all,
  jbasis_eval_all_with_grad
)
import thread_control
import nlopt
import matplotlib.pyplot as plt
import time

def max_offdiag(J):
    """
    Compute max absolute off-diagonal entry across all matrices J[:,:,ell].

    Parameters
    ----------
    J : ndarray, shape (N, N, D)

    Returns
    -------
    float
        Maximum |J[i,j,ell]| for i!=j.
    """
    N, N2, D = J.shape
    assert N == N2
    mask = ~np.eye(N, dtype=bool)
    # max over each matrix, then over ell
    return np.max(np.abs(J[mask, :]))


def compute_joint_spectrum(D, n, kappa,
                           tol=1e-10,
                           max_sweeps=50):
  """
  Build D Jacobi matrices for total degree n, jointly diagonalize them,
  and return the matched eigenvalues as an array of shape (N, D).

  Parameters
  ----------
  D : int
      Dimension of the simplex.
  n : int
      Maximum total degree for the Jacobi basis (Pi_n^D).
  kappa : array_like, shape (D+1,)
      Jacobi parameters on the simplex.
  tol : float
      Threshold on |s| for JEV D (Cardoso–Souloumiac).
  max_sweeps : int
      Maximum sweeps in the JEV D algorithm.

  Returns
  -------
  eigs : ndarray, shape (N, D)
      Joint eigenvalue tuples (lambda_1, ..., lambda_D) per basis vector.
      Each row corresponds to one joint eigenvector (i.e., one column of V).
  J_diag : ndarray, shape (N, N, D)
      The approximately diagonalized Jacobi matrices.
  V : ndarray, shape (N, N)
      Joint diagonalizer (columns are joint eigenvectors).
  info : int
      JEV D info flag: 0 = converged, 1 = not converged, <0 = input error.
  sweeps : int
      Number of sweeps actually performed.
  max_offdiag : float
      Last max |s| reported by the algorithm.
  """

  kappa = np.asarray(kappa, dtype=np.float64)
  if kappa.shape[0] != D + 1:
    raise ValueError("kappa must have length D+1")

  # 1) Build Jacobi matrices for each coordinate x_i, i=1..D.
  #    Assume jmat_build returns an array J_all with shape (D, N, N),
  #    where N = dim Pi_n^D and each J_all[i,:,:] is the Jacobi
  #    matrix for coordinate x_{i+1}.
  J_all = jmat_build(D, n, kappa)   # shape (D, N, N)
  if J_all.ndim != 3 or J_all.shape[0] != D:
    raise RuntimeError("jmat_build returned unexpected shape")

  D_, N, N2 = J_all.shape
  if N != N2:
    raise RuntimeError("Jacobi matrices must be square")
  if D_ != D:
    raise RuntimeError("First dimension of J_all must be D")

  # 2) Pack into shape (N, N, D) (AoS stack of matrices) for jevd_serial.
  #    Each slice J_stack[:, :, ell] is J_{ell}, and we ensure Fortran
  #    layout so it matches the C layout J[i + m*(j + nn*m)].
  J_stack = np.asfortranarray(J_all.transpose(1, 2, 0))  # (N, N, D), F-order

  # 3) Run serial JEV D to jointly diagonalize (N x N x D) stack.
  J_diag, V, info, sweeps, max_offdiag_jevd = jevd_serial(
    J_stack,
    tol=tol,
    max_sweeps=max_sweeps,
    accumulate_V=True
  )
  # J_diag has shape (N, N, D), V is N x N (both Fortran).

  # 4) Extract eigenvalues per matrix (the diagonals).
  #    For each dimension ell = 0..D-1, take diag(J_diag[:,:,ell]).
  eig_list = []
  for ell in range(D):
    diag_ell = np.diag(J_diag[:, :, ell])
    eig_list.append(diag_ell)

  # Stack columns → shape (N, D)
  eigs = np.stack(eig_list, axis=1)

  # Optional: sort the joint eigenvalues for convenience
  # (global ordering is otherwise arbitrary but consistent across dims).
  # Here we sort lexicographically by the tuple (lambda_1, ..., lambda_D).
  sort_idx = np.lexsort(eigs.T[::-1])  # last key is first column
  eigs_sorted = eigs[sort_idx, :]
  # 5) Compute final off-diagonal residual
  final_offdiag = max_offdiag(J_diag)


  return eigs_sorted, J_diag, V, info, sweeps, max_offdiag_jevd, final_offdiag


def build_initial_guess(D, n_basis, kappa, tol=1e-12, max_sweeps=10):
  """
  Use joint spectrum as node seeds and a least-squares system
  to get an initial weight guess.

  Returns
  -------
  X0 : ndarray, shape (N, D)
  w0 : ndarray, shape (N,)
  z0 : ndarray, shape (N*(D+1),)
  """
  # Joint spectrum from Jacobi matrices
  eigs, J_diag, V_joint, info, sweeps, max_off, final_offdiag = compute_joint_spectrum(
    D, n_basis, kappa, tol=tol, max_sweeps=max_sweeps
  )

  X0 = eigs  # N x D nodes
  N = X0.shape[0]

  # Build basis structures
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n_basis, kappa)
  M = alpha_table.shape[0]

  # Vandermonde at these nodes
  V = jbasis_eval_all(X0, kappa, n_basis, alpha_table, tail_deg, inv_h, D)

  # Desired integrals = e1
  e1 = np.zeros(M)
  e1[0] = 1.0

  # Least-squares for initial weights
  # V^T w ≈ e1  =>  minimize ||V^T w - e1||^2
  # Use normal eqs or lstsq on V.
  w0, *_ = np.linalg.lstsq(V.T, e1, rcond=None)

  # Pack z0
  z0 = np.concatenate([X0.reshape(-1), w0])
  return z0, alpha_table, tail_deg, inv_h, e1

# Central finite-difference gradient (2nd order accurate)
def finite_diff_grad(f, x, h):
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  grad = np.empty_like(x)
  # Could be expensive: O(nvar) function calls per gradient
  fx = f(x)  # not strictly needed for central diff, but cached if desired
  for i in range(x.size):
    xp = x.copy()
    xm = x.copy()
    xp[i] += h
    xm[i] -= h
    fp = f(xp)
    fm = f(xm)
    grad[i] = (fp - fm) / (2.0 * h)
  return grad

def optimize_quadrature(D,
                        n_nodes,      # N = number of nodes
                        m_basis,      # degree for basis Pi_m^D
                        kappa,
                        z0,
                        algo=nlopt.LD_SLSQP,   # gradient-based by default
                        maxeval=5000,
                        tol_f=1e-12,
                        fd_step=1e-6,
                        verbose=True):
  """
  Optimize node/weight vector z to minimize

      f(z) = 0.5 ||F(z)||^2,

  where
      F(z) = V^T w - e1,  V_{pj} = P_j(x_p),
      P_j = orthonormal Jacobi basis on T^D up to degree m_basis.

  Unknowns:
    z in R^{N*(D+1)} = [X_flat, w_flat],  N = n_nodes.

  Layout:
    - First N*D entries: X_flat, reshaped to (N, D) with rows x_p
      (barycentric coords, last coord = 1 - sum(x_p))
    - Last N entries:    w_flat = weights

  Constraints:
    For each node p:
      (i)   x_{p,i} >= 0                       (non-negativity)
      (ii)  sum_i x_{p,i} <= 1                 (simplex constraint)
    For weights:
      (iii) w_p >= 0
      (iv)  sum_p w_p = 1                      (normalized measure)

  Gradients:
    - Objective: central finite differences (2nd order accurate).
    - Constraints: analytical gradients.

  Parameters
  ----------
  D : int
      Dimension of the simplex.
  n_nodes : int
      Number of nodes N.
  m_basis : int
      Degree m for basis Pi_m^D used in F(z).
  kappa : array_like, shape (D+1,)
      Jacobi parameters.
  z0 : array_like, shape (N*(D+1),)
      Initial guess [X_flat, w_flat].
  algo : nlopt algorithm enum
      Must be a gradient-based algorithm to exploit the gradients, e.g.
      nlopt.LD_SLSQP, nlopt.LD_MMA, etc.
  maxeval : int
      Max number of objective evaluations.
  tol_f : float
      Relative tolerance on f.
  fd_step : float
      Finite-difference step h for central differences.
  verbose : bool
      If True, print occasional diagnostics.

  Returns
  -------
  z_opt : ndarray, shape (N*(D+1),)
      Optimized decision vector.
  f_opt : float
      Final objective value 0.5 ||F(z_opt)||^2.
  """
  kappa = np.asarray(kappa, dtype=np.float64)
  if kappa.shape[0] != D + 1:
    raise ValueError("kappa must have length D+1")

  z0 = np.asarray(z0, dtype=np.float64)
  N = n_nodes
  nvar = N * (D + 1)
  if z0.size != nvar:
    raise ValueError(f"z0 must have length N*(D+1)={nvar}, got {z0.size}")

  # Build basis structures ONCE for degree m_basis
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, m_basis, kappa)
  M = alpha_table.shape[0]  # dim Pi_m^D

  e1 = np.zeros(M, dtype=np.float64)
  e1[0] = 1.0

  # Helper to unpack z -> X, w
  def unpack_z(z):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    X_flat = z[:N * D]
    W_flat = z[N * D:]
    X = X_flat.reshape(N, D)
    w = W_flat
    return X, w

  # Vector residual F(z) = V^T w - e1
  def F_vec(z):
    X, w = unpack_z(z)
    V = jbasis_eval_all(X, kappa, m_basis,
                        alpha_table, tail_deg, inv_h, D)
    Ihat = V.T @ w
    return Ihat - e1

  # Pure scalar objective f(z) without gradient
  def f_scalar(z):
    F = F_vec(z)
    return 0.5 * float(F @ F)



  # Simple eval counter for diagnostics
  eval_count = {"n": 0}

  """
  # Objective wrapper in NLopt style with FD gradient
    def objective(z, grad):
      if grad.size > 0:
        # Fill gradient via central finite differences
        g = finite_diff_grad(f_scalar, z, fd_step)
        grad[:] = g[:]
      fval = f_scalar(z)
      eval_count["n"] += 1
      if verbose and eval_count["n"] % 50 == 0:
        print(f"[objective] eval {eval_count['n']}, f(z) = {fval:.6e}")
      return fval
  """
  # Objective wrapper in NLopt style with analytical gradient
  def objective(z, grad):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    X, w = unpack_z(z)

    if grad.size > 0:
      # Evaluate basis and gradients
      V, dV = jbasis_eval_all_with_grad(
        X, kappa, m_basis,
        alpha_table, tail_deg, inv_h, D
      )  # V: (N,M), dV: (N,M,D)
    else:
      V = jbasis_eval_all(
        X, kappa, m_basis,
        alpha_table, tail_deg, inv_h, D
      )
      dV = None

    Ihat = V.T @ w             # (M,)
    F = Ihat - e1              # (M,)

    fval = 0.5 * float(F @ F)

    if grad.size > 0:
      # Gradient ∇f = J^T F, where J is Jacobian of F.

      # Node part: z[0 : N*D]
      # For each p, the derivative wrt x_{p,:} is:
      #   ∂f/∂x_{p,:} = w[p] * dV[p,:,:]^T @ F  ∈ R^D
      grad_nodes = np.zeros((N, D), dtype=np.float64)
      for p in range(N):
        # dV[p,:,:]: (M,D), F: (M,) -> (D,)
        grad_nodes[p, :] = w[p] * (dV[p, :, :].T @ F)

      grad[:N*D] = grad_nodes.reshape(-1)

      # Weight part: z[N*D : N*(D+1)]
      # ∂F/∂w_p = P(x_p) = V[p,:], so:
      #   ∂f/∂w_p = F · V[p,:]  => (V @ F)[p]
      grad[N*D:] = V @ F

    eval_count["n"] += 1
    if verbose and eval_count["n"] % 50 == 0:
      print(f"[objective] eval {eval_count['n']}, f(z) = {fval:.6e}")
    return fval


  m_ineq = N * (D + 2)   # N*D (X>=0) + N (sum X <= 1) + N (w>=0)
  nvar   = (D + 1) * N   # variables: N*D coords + N weights
  
  def ineq_constraints(result, z, grad):
      """
      g(z) ∈ R^{N*(D+1)} s.t. g(z) <= 0 encodes:
  
        (i)  -X[p,i] <= 0           (X >= 0)
        (ii) sum_i X[p,i] - 1 <= 0  (simplex row)
        (iii)-w[p] <= 0             (w >= 0)
      """
      z = np.asarray(z, dtype=np.float64)
      X = z[:N*D].reshape(N, D)
      w = z[N*D:]
      # Fill result
      k = 0
  
      # (i) -X[p,i]
      result[k : k + N*D] = -X.ravel()
      k += N*D
  
      # (ii) sum_i X[p,i] - 1
      result[k : k + N] = X.sum(axis=1) - 1.0
      k += N
  
      # (iii) -w[p]
      result[k : k + N] = -w
  
      # Gradients: grad has shape (m_ineq, nvar)
      if grad.size > 0:
          grad[:] = 0.0
          # (i) d(-X[p,i])/dX[p,i] = -1
          k = 0
          for p in range(N):
              for i in range(D):
                  con_idx = k                   # which constraint
                  var_idx = p*D + i             # index of X[p,i] in z
                  grad[con_idx, var_idx] = -1.0
                  k += 1
  
          # (ii) d(sum_i X[p,i] - 1)/dX[p,i] = 1
          for p in range(N):
              con_idx = k
              for i in range(D):
                  var_idx = p*D + i
                  grad[con_idx, var_idx] = 1.0
              k += 1
  
          # (iii) d(-w[p])/dw[p] = -1
          for p in range(N):
              con_idx = k
              var_idx = N*D + p                # weight index
              grad[con_idx, var_idx] = -1.0
              k += 1
  
  
  def eq_constraints(result, z, grad):
      """
      h(z) ∈ R^1 s.t. h(z) = 0 encodes:
         sum_p w[p] = 1
      """
      z = np.asarray(z, dtype=np.float64)
      w = z[N*D:]
  
      result[0] = np.sum(w) - 1.0
  
      if grad.size > 0:
          grad[:] = 0.0
          # grad shape (1, nvar)
          grad[0, N*D : N*(D+1)] = 1.0

  # === NLopt setup ===

  opt = nlopt.opt(algo, nvar)
  opt.set_min_objective(objective)
  opt.set_maxeval(maxeval)
  opt.set_xtol_rel(1e-7)
  opt.set_ftol_rel(tol_f)
  # tolerances per constraint
  ineq_tol = np.full(m_ineq, 1e-14, dtype=np.float64)
  eq_tol   = np.array([1e-14], dtype=np.float64)
  opt.add_inequality_mconstraint(ineq_constraints, ineq_tol)
  opt.add_equality_mconstraint(eq_constraints, eq_tol)


  print(opt.get_initial_step(z0))

  # === Run optimization ===

  try:
    z_opt = opt.optimize(z0)
    f_opt = opt.last_optimum_value()
    status = opt.last_optimize_result()
  except nlopt.RoundoffLimited:
    z_opt = z0
    f_opt = obj(z_opt, np.zeros_like(z_opt))
    status = nlopt.ROUNDING_ERRORS

  # Info
  status_map = {
    nlopt.SUCCESS:       "Success",
    nlopt.STOPVAL_REACHED: "Stopval reached",
    nlopt.FTOL_REACHED:  "Function tolerance reached",
    nlopt.XTOL_REACHED:  "X tolerance reached",
    nlopt.MAXEVAL_REACHED: "Max eval reached",
    nlopt.MAXTIME_REACHED: "Max time reached",
    nlopt.FAILURE:       "Failure",
    nlopt.INVALID_ARGS:  "Invalid args",
    nlopt.OUT_OF_MEMORY: "Out of memory",
    nlopt.ROUNDOFF_LIMITED: "Roundoff limited",
    nlopt.FORCED_STOP:   "Forced stop",
  }
  msg = status_map.get(status, f"Unknown status {status}")

  info = {"status": int(status), "message": msg}

  if verbose:
    print(f"NLopt finished with status {status}: {msg}")
    print(f"Optimization finished after {eval_count['n']} objective evals.")
    print(f"Final f(z_opt) = {f_opt:.16e}")

    # Sanity diagnostics
    X_opt, w_opt = unpack_z(z_opt)
    V_opt = jbasis_eval_all(X_opt, kappa, m_basis,
                            alpha_table, tail_deg, inv_h, D)
    Ihat_opt = V_opt.T @ w_opt
    F_opt = Ihat_opt - e1
    print("||Ihat - e1||_2 =", np.linalg.norm(F_opt))
    print("sum(weights)     =", np.sum(w_opt))
    print("min(weights)     =", np.min(w_opt))

  return z_opt, float(f_opt)

def finite_diff_jacobian(F, x, h):
  """
  Central 2nd-order finite-difference Jacobian for a vector-valued F.

  Parameters
  ----------
  F : callable
    F(x) -> array_like, shape (M,)
  x : array_like, shape (nvar,)
    Point at which to evaluate the Jacobian.
  h : float
    Step size for central differences.

  Returns
  -------
  J : ndarray, shape (M, nvar)
    Approximate Jacobian, J[i,j] = dF_i/dx_j.
  """
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  f0 = np.asarray(F(x), dtype=np.float64)
  M = f0.size
  nvar = x.size

  J = np.empty((M, nvar), dtype=np.float64)

  for j in range(nvar):
    x[j] += h
    fp = np.asarray(F(x), dtype=np.float64)
    x[j] -= 2*h
    fm = np.asarray(F(x), dtype=np.float64)
    x[j] += h
    J[:, j] = (fp - fm) / (2.0 * h)

  return J

def project_simplex_leq1(y):
  """
  Project y in R^D onto { x >= 0, sum(x) <= 1 } w.r.t. Euclidean norm.
  """
  y = np.asarray(y, dtype=np.float64)
  y_plus = np.clip(y, 0.0, None)
  s = y_plus.sum()
  if s <= 1.0:
    return y_plus

  # Project onto standard simplex {x >= 0, sum x = 1}
  # Algorithm: sort, find threshold theta, set x_i = max(y_i - theta, 0)
  u = np.sort(y)[::-1]        # descending
  cssv = np.cumsum(u)
  rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1.0))[0][-1]
  theta = (cssv[rho] - 1.0) / (rho + 1.0)
  x = np.clip(y - theta, 0.0, None)
  return x

def project_z(z, D, N):
  """
  Project z = [X_flat, w] onto:
      X_{p,:} in {x>=0, sum x <=1}, w_p >= 0.
  """
  z = np.asarray(z, dtype=np.float64).reshape(-1)
  X_flat = z[:N*D]
  w = z[N*D:]

  X = X_flat.reshape(N, D)
  for p in range(N):
    X[p, :] = project_simplex_leq1(X[p, :])

  w = np.clip(w, 0.0, None)

  z_proj = np.concatenate([X.reshape(-1), w])
  return z_proj


def gauss_newton(F, J, z0, data=None,
                 alpha=0.1,
                 tol=1e-16,
                 tol_up=1e3,
                 maxiter=1000,
                 verbose=True):
  """
  Gauss-Newton iteration:
      z_{k+1} = z_k - alpha * (J^+ F)(z_k)

  F : callable
    F(z, data) -> (M,) residual.
  J : callable
    J(z, data) -> (M, nvar) Jacobian.
  """
  z = np.asarray(z0, dtype=np.float64).reshape(-1)
  Fk = F(z, data)
  pk = np.linalg.norm(Fk, 2)

  iter_count = 0
  converged = False

  if verbose:
    print(f"GN iter {iter_count:4d}, ||F||_2 = {pk:.6e}")

  while pk > tol and pk < tol_up and iter_count < maxiter:
    iter_count += 1

    Jk = np.asarray(J(z, data), dtype=np.float64)  # M x nvar

    # Solve Jk * delta ≈ Fk in least-squares sense
    delta, *_ = np.linalg.lstsq(Jk, Fk, rcond=None)

    # Step
    z_new = z - alpha * delta

    # project onto feasible set C
    z = project_z(z_new, D, N)

    # New residual
    Fk = F(z, data)
    pk = np.linalg.norm(Fk, 2)

    if verbose:
      print(f"GN iter {iter_count:4d}, ||F||_2 = {pk:.6e}")

    if pk <= tol:
      converged = True
      break
    if pk >= tol_up:
      break

  info = {
    "pk": float(pk),
    "iter": int(iter_count),
    "converged": bool(converged),
  }
  return z, info


def optimize_quadrature_gn(D,
                           n_nodes,      # N = number of nodes
                           m_basis,      # degree for basis Pi_m^D
                           kappa,
                           z0,
                           alpha=0.1,
                           maxiter=100,
                           fd_step=1e-6,
                           tol=1e-15,
                           tol_up=1e3,
                           verbose=True):
  """
  Gauss-Newton optimizer for the quadrature system
      F(z) = V^T w - e1 = 0.

  Uses finite-difference Jacobian via finite_diff_jacobian and
  the generic gauss_newton stepper.
  """
  kappa = np.asarray(kappa, dtype=np.float64)
  if kappa.shape[0] != D + 1:
    raise ValueError("kappa must have length D+1")

  z0 = np.asarray(z0, dtype=np.float64)
  N = n_nodes
  nvar = N * (D + 1)
  if z0.size != nvar:
    raise ValueError(f"z0 must have length N*(D+1)={nvar}, got {z0.size}")

  # Build basis structures once
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, m_basis, kappa)
  M = alpha_table.shape[0]

  e1 = np.zeros(M, dtype=np.float64)
  e1[0] = 1.0

  # helper: unpack z -> X, w
  def unpack_z(z):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    X_flat = z[:N * D]
    W_flat = z[N * D:]
    X = X_flat.reshape(N, D)
    w = W_flat
    return X, w

  # vector residual F(z)
  def F_vec(z, data=None):
    X, w = unpack_z(z)
    V = jbasis_eval_all(X, kappa, m_basis,
                        alpha_table, tail_deg, inv_h, D)  # N x M
    Ihat = V.T @ w
    F = Ihat - e1
    return F

  # Analytic Jacobian via basis gradients
  def J_a(z, data=None):
    """
    Analytic Jacobian J(z) of F_vec(z):

      F(z) = V(X)^T w - e1,    F ∈ R^M,
      z = [X_flat, w] ∈ R^{N*(D+1)}.

    Structure:
      dF/dx_{p,ell} = w[p] * dP(x_p)/dx_ell,
      dF/dw_p       = P(x_p).
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    X, w = unpack_z(z)               # X: (N,D), w: (N,)

    # V: (N,M), dV: (N,M,D)
    V, dV = jbasis_eval_all_with_grad(
      X, kappa, m_basis,
      alpha_table, tail_deg, inv_h, D
    )

    J = np.zeros((M, nvar), dtype=np.float64)

    # Node part: columns 0 .. N*D-1
    # For each node p, dF/dx_{p,:} is an M×D block: w[p] * dV[p,:,:]
    for p in range(N):
      # dV[p,:,:] has shape (M,D); we want J[:, p*D:(p+1)*D]
      J[:, p*D:(p+1)*D] = w[p] * dV[p, :, :]

    # Weight part: columns N*D .. N*(D+1)-1
    # dF/dw_p = P(x_p) = V[p,:]
    for p in range(N):
      J[:, N*D + p] = V[p, :]

    return J

  # Jacobian via finite differences
  def J_fd(z, data=None):
    return finite_diff_jacobian(F_vec, z, fd_step)

  if verbose:
    print(f"[GN] dim Pi_{m_basis}^{{{D}}} = M = {M}, N = {N}, nvar = {nvar}")
    F0 = F_vec(z0)
    print(f"[GN] initial ||F||_2 = {np.linalg.norm(F0):.6e}")

  # Run Gauss-Newton
  z_opt, info = gauss_newton(F_vec, J_a, z0,
                             data=None,
                             alpha=alpha,
                             tol=tol,
                             tol_up=tol_up,
                             maxiter=maxiter,
                             verbose=verbose)

  if verbose:
    print(f"[GN] finished: converged={info['converged']}, "
          f"iter={info['iter']}, ||F||_2={info['pk']:.6e}")

    X_opt, w_opt = unpack_z(z_opt)
    V_opt = jbasis_eval_all(X_opt, kappa, m_basis,
                            alpha_table, tail_deg, inv_h, D)
    Ihat_opt = V_opt.T @ w_opt
    F_opt = Ihat_opt - e1
    print("[GN] diagnostics:")
    print("  ||Ihat - e1||_2 =", float(np.linalg.norm(F_opt)))
    print("  sum(weights)     =", float(np.sum(w_opt)))
    print("  min(weights)     =", float(np.min(w_opt)))
    print("  max(weights)     =", float(np.max(w_opt)))

  # For consistency with optimize_quadrature, return z_opt, scalar f
  f_opt = 0.5 * float(info["pk"] ** 2)
  return np.asarray(z_opt, dtype=np.float64), f_opt, info, V_opt

def cond_number(V):
  """
  Return the 2-norm condition number of V.
  """
  # singular values
  s = np.linalg.svd(V, compute_uv=False)
  return s[0] / s[-1]

if __name__ == "__main__":
# Example settings

  thread_control.set_omp_threads(8)  

  start_time = time.time()
  
  D = 3
  n_node_deg = 11
  m_basis = 16
  #kappa = np.array([0.5, 0.5, 0.5], dtype=np.float64)
  kappa = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

  # 1) Build initial guess from joint spectrum
  z0, alpha_table, tail_deg, inv_h, e1 = build_initial_guess(D, n_node_deg, kappa,
                                                                     tol=1e-14, max_sweeps=100)
  N = (z0.shape[0]) // (D+1)
  print("Initial N nodes:", N)
  print("Initial f(z0) evaluation...")
  print("Initial sum weights:", np.sum(z0[N*D::]))


  z_opt, f_opt = optimize_quadrature(D,
                      N,      # N = number of nodes
                      m_basis,      # degree for basis Pi_m^D
                      kappa,
                      z0,
                      algo=nlopt.LD_SLSQP,   # gradient-based by default
                      maxeval=5000,
                      tol_f=1e-12,
                      fd_step=1e-6,
                      verbose=True)
  
  #z_opt = np.loadtxt("tetquad_10_16_curr.txt")
  z_optn, f_optn, infon, V_opt = optimize_quadrature_gn(
    D,
    N,
    m_basis,
    kappa,
    z_opt,
    alpha=1,
    maxiter=4000,
    fd_step=1e-6,
    tol=1e-15,
    tol_up=1e3,
    verbose=True,
  )

  z_opt = z_optn
  f_opt = f_optn
  condV = cond_number(V_opt)
  end_time = time.time()
  elapsed_time = end_time - start_time  
  print("Final objective f(z_opt) =", f_opt)
  print(f"Elapsed time: {elapsed_time:.4f} seconds")
  print("Condition number of interp op on abscissa: ", condV)

  # 3) Unpack solution
  X_opt = z_opt[:N*D].reshape(N, D)
  w_opt = z_opt[N*D:]

  fig = plt.figure()
  if D == 3: 
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_opt[:,0], X_opt[:,1], X_opt[:,2])
    fname = "tetquad.txt"
  elif D== 2:
    ax = fig.add_subplot()
    ax.scatter(X_opt[:,0], X_opt[:,1])
    x_coords = [0, 1, 0, 0]
    y_coords = [0, 0, 1, 0] 
    ax.plot(x_coords, y_coords) # 'o' adds markers to the vertices
    fname = "triquad.txt"

  plt.show()
  np.savetxt(fname, z_opt, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)


  #D = 3
  #n = 10
  ##kappa = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
  #kappa = np.array([0.5, 0.7, 1.8, 2.5], dtype=np.float64)
  #D = 2
  #n = 10
  ##kappa = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
  #kappa = np.array([0.5, 0.7, 1.8], dtype=np.float64)

  #eigs, J_diag, V, info, sweeps, max_off, final_offdiag = compute_joint_spectrum(
  #  D, n, kappa,
  #  tol=1e-12,
  #  max_sweeps=10
  #)

  #print("info       =", info)
  #print("sweeps    =", sweeps)
  #print("max_off   =", max_off)
  #print("final_offdiag(max|J|)   =", final_offdiag)  
  #print("eigs shape:", eigs.shape)

 
  #fig = plt.figure()
  #if D == 3: 
  #  ax = fig.add_subplot(projection='3d')
  #  ax.scatter(eigs[:,0], eigs[:,1], eigs[:,2])
  #else:
  #  ax = fig.add_subplot()
  #  ax.scatter(eigs[:,0], eigs[:,1])
  #plt.show()
