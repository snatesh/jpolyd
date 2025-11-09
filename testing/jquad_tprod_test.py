import numpy as np
from jquad_mapped import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

D = 3
ns = np.arange(1,20)
I_true = 1./3080.
errs = np.zeros(len(ns))
for j in range(len(ns)):
  n = ns[j]
  X, W = jquad_mapped_build(D,n)
  f_vals = X[:,0]**19 + X[:,1]**19 + X[:,2]**19
  I_approx = np.dot(W, f_vals)
  errs[j] = np.abs(I_approx - I_true)/I_true
  print(errs)

fig, ax = plt.subplots() 
ax.semilogy(ns, errs,linestyle=':', marker='o', markersize=6, color='blue')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()





