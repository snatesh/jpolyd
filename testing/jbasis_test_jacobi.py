import numpy as np
from jquad_tprod import *
from jweight import *
from jbasis import *
import matplotlib.pyplot as plt 

#D = 2
#n = 5
#nquad = 20
#kappa = np.array([1.7, 3.3, 2.8], dtype=np.float64)
#D = 4
#n = 10
#nquad = 10
#kappa = np.array([2.5, 1.5, 5.5, 3.5, 6.5])

#D = 4
#n = 10
#nquad = n+1
#kappa = np.array([2.5, 1.5, 5.5, 3.5, 6.5])

D = 5
n = 10
nquad = n+1
kappa = np.array([2.5, 1.5, 5.5, 3.5, 6.5, 1.7])

# Build basis structures once
alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)

# Build mapped tensor-product quad rule on simplex (unit weight)
points, weights = jquad_mapped_build_kappa(D, nquad, kappa) 

print("sum(weights) =", weights.sum())  # expect ~ 1.0


# Evaluate basis at quadrature points
V = jbasis_eval_all(points, kappa, n, alpha_table, tail_deg, inv_h, D)


npts, M = V.shape
print(f"V shape = {V.shape}, npts = {npts}, M = {M}")

# Form Gram matrix G = V^T diag(w_norm) V
W = weights[:, None]         # npts x 1
VW = V * W                  # npts x M, row-weighted
G = V.T @ VW                # M x M

# Compare to identity
I = np.eye(M)
E = G - I

frob_err = np.linalg.norm(E, ord="fro")
max_err = np.max(np.abs(E))

print("Gram matrix deviation ||G - I||_F =", frob_err)
print("Max absolute entry of G - I      =", max_err)

spy_matrix = (np.abs(G) > 1e-6)
plt.figure(figsize=(5, 5))
plt.spy(spy_matrix, markersize=5)
plt.show()
