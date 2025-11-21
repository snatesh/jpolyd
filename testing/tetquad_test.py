import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

D = 3
z_opt = np.loadtxt("tetquad_10_16.txt", delimiter=' ', comments='# ').reshape(-1)

nvar = z_opt.size
N = nvar // (D + 1)
assert nvar == (D + 1) * N, "File length not compatible with D+1 variables per node"

X_flat = z_opt[:N*D]
W = z_opt[N*D:]
X = X_flat.reshape(N, D)

print(W)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])

deg = 15
I_true = 3.0 / ((deg+1)*(deg+2)*(deg+3))
f_vals = X[:,0]**deg + X[:,1]**deg + X[:,2]**deg
I_approx = np.dot(W, f_vals)
print(np.abs(I_approx/6.0-I_true))
print(I_true)
print(I_approx)
print(I_true/I_approx)
plt.show()
