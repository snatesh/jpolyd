import numpy as np
from jquad_tprod import *
from jweight import *
from jbasis import *
import matplotlib.pyplot as plt 

D = 4
n = 10
nquad = 30
kappa = np.array([2.5, 1.5, 5.5, 3.5, 6.5])
#kappa = np.array([0.5, 0.5, 1.5])

D = 2
n = 5
nquad = 20
kappa = np.array([2.5, 1.5, 5])


# Build basis structures once
alpha_table, tail_deg, inv_h = jbasis_build_structures(D, n, kappa)

# Build mapped tensor-product quad rule on simplex (unit weight)
points, weights = jquad_mapped_build(D, nquad)  # your existing mapped quad

# Check what the rule integrates for constant 1
Wk = jweight_eval(points, D, 1, len(points), kappa, D)

# Renormalize weights so they represent the normalized measure dw
w_norm = weights * Wk

# Evaluate basis at quadrature points
V = jbasis_eval_all(points, kappa, n, alpha_table, tail_deg, inv_h, D)


npts, M = V.shape
print(f"V shape = {V.shape}, npts = {npts}, M = {M}")

# Form Gram matrix G = V^T diag(w_norm) V
W = w_norm[:, None]         # npts x 1
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
