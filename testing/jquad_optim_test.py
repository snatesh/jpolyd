from jquad_optim import *
import thread_control
import matplotlib.pyplot as plt
import math

def dim_Pi(D, n):
  """Dimension of full total-degree ≤ n polynomials on R^D."""
  return math.comb(n + D, D)

def make_fname(D, node_deg, m_basis, kappa):
  """
  Example filename:
    t3_kappa0.5-0.5-0.5-0.5_N286_M969_n10_m16.txt
  """
  # convert kappa vector to string "0.5-0.5-0.5-0.5"
  kappa_str = "-".join([f"{float(x):.6g}" for x in kappa])

  # compute N, M
  N = dim_Pi(D, node_deg)
  M = dim_Pi(D, m_basis)

  fname = (
    f"t{D}_"
    f"kappa{kappa_str}_"
    f"N{N}_M{M}_"
    f"n{node_deg}_m{m_basis}.txt"
  )
  return fname

thread_control.set_omp_threads(8)  

D = 3
n_node_deg = 11
m_basis = 16
kappa = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
#kappa = np.array([0.5, 2.5, 1.7, 3.2], dtype=np.float64)
X, w, V_opt, status = optimize_quadrature(
  D,
  n_node_deg,
  m_basis,
  kappa,
  z0=None,
  want_V_opt=True,
  max_nlopt_eval = 10000,
  max_gn_iter = 4000,
  gn_step = 1,
  tol = 5e-15,
  tol_up = 1e3,
  verbose = True,
)

fig = plt.figure()
if D == 3: 
  ax = fig.add_subplot(projection='3d')
  ax.scatter(X[:,0], X[:,1], X[:,2])
elif D== 2:
  ax = fig.add_subplot()
  ax.scatter(X[:,0], X[:,1])
  x_coords = [0, 1, 0, 0]
  y_coords = [0, 0, 1, 0] 
  ax.plot(x_coords, y_coords) # 'o' adds markers to the vertices


fname = make_fname(D, n_node_deg, m_basis, kappa)
print("Saving to:", fname)
np.savetxt(fname, np.concatenate([X.reshape(-1), w]), fmt="%.18e")

plt.show()
