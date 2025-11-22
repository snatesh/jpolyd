import numpy as np
from jbasis import (
  jbasis_build_structures, 
  jbasis_eval_all,
  jbasis_eval_all_with_grad
)
import thread_control
import matplotlib.pyplot as plt




def verify_derivatives(D=2, N=6, m_basis=4, seed=1):
  """
  Compare finite-difference and analytic derivatives for:
      F(z)  (vector residual)
      f(z) = 0.5 ||F||^2  (scalar objective)
  
  For decreasing FD step sizes h, plot/print relative errors.
  """
  
  np.random.seed(seed)
  # 1. Setup random z = (X,w) with X in simplex
  # Sample barycentric coordinates in R^{D+1}, normalize, then drop last coord.
  U = np.random.rand(N, D+1)
  Y = U / np.sum(U, axis=1, keepdims=True)  # each row sums to 1
  X = Y[:, :D].copy()                        # last coord is implicit = 1 - sum(X)
  
  # random positive weights normalized to sum 1
  w = np.random.rand(N)
  w /= np.sum(w)
  
  z = np.concatenate([X.reshape(-1), w])  

  
  # 2. Build basis data structures
  kappa = np.ones(D+1)*0.5
  alpha_table, tail_deg, inv_h = jbasis_build_structures(D, m_basis, kappa)
  M = alpha_table.shape[0]
  nvar = N*(D+1)

  def unpack_z(z):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    X_flat = z[:N * D]
    W_flat = z[N * D:]
    X = X_flat.reshape(N, D)
    w = W_flat
    return X, w

  
  # --- Vector residual and analytic Jacobian ---
  e1 = np.zeros(M, dtype=np.float64)
  e1[0] = 1.0
 
  def F_vec(z):
    X, w = unpack_z(z)
    V = jbasis_eval_all(X, kappa, m_basis,
                        alpha_table, tail_deg, inv_h, D)
    Ihat = V.T @ w
    return Ihat - e1
  
  def J_a(z):
    z = np.asarray(z, float)
    X, w = unpack_z(z)
    V, dV = jbasis_eval_all_with_grad(
        X, kappa, m_basis,
        alpha_table, tail_deg, inv_h, D
    )
    J = np.zeros((M, nvar))
    for p in range(N):
      J[:, p*D:(p+1)*D] = w[p] * dV[p,:,:]
    for p in range(N):
      J[:, N*D + p] = V[p, :]
    return J
  
  # --- Scalar objective and its analytic gradient ---
  
  def f_scalar(z):
    F = F_vec(z)
    return 0.5 * float(F @ F)
  
  def grad_a(z):
    z = np.asarray(z, float)
    X, w = unpack_z(z)
    V, dV = jbasis_eval_all_with_grad(
      X, kappa, m_basis,
      alpha_table, tail_deg, inv_h, D
    )
    F = F_vec(z)
    grad = np.zeros(nvar)
  
    # node part
    for p in range(N):
      grad[p*D:(p+1)*D] = w[p] * (dV[p,:,:].T @ F)
    
    # weight part
    grad[N*D:] = V @ F
    
    return grad
  
  # --- Finite-difference helpers ---
  
  def FD_grad(vec_func, z, h):
    # directional FD for a vector-valued function F
    z = np.asarray(z, float)
    J = np.zeros((len(vec_func(z)), len(z)))
    for k in range(len(z)):
      zp = z.copy(); zp[k] += h
      zm = z.copy(); zm[k] -= h
      J[:,k] = (vec_func(zp) - vec_func(zm)) / (2*h)
    return J
  
  def FD_grad_scalar(scalar_func, z, h):
    z = np.asarray(z, float)
    grad = np.zeros_like(z)
    for k in range(len(z)):
      zp = z.copy(); zp[k] += h
      zm = z.copy(); zm[k] -= h
      grad[k] = (scalar_func(zp) - scalar_func(zm)) / (2*h)
    return grad
  
  # --- Convergence test ---
  hs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3,
        1e-4, 5e-5, 1e-5, 5e-6, 1e-6,
        5e-7, 1e-7, 5e-8]
  
  J_exact = J_a(z)
  g_exact = grad_a(z)
  
  EJ = np.zeros(len(hs))
  Eg = np.zeros(len(hs))
  idx = 0
  for h in hs:
    J_fd = FD_grad(F_vec, z, h)
    g_fd = FD_grad_scalar(f_scalar, z, h)
    
    EJ[idx] = np.linalg.norm(J_fd - J_exact) / np.linalg.norm(J_exact)
    Eg[idx] = np.linalg.norm(g_fd - g_exact) / np.linalg.norm(g_exact)
    idx += 1

  return EJ, Eg, hs

thread_control.set_omp_threads(10)
Ej, Eg, hs = verify_derivatives(D=2, N=6, m_basis=4, seed=1)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 8)) 
ax[0].loglog(hs, Ej)
ax[0].set_title('Jacobian')
ax[1].loglog(hs, Eg)
ax[1].set_title('Gradient')
plt.tight_layout()
plt.show()

