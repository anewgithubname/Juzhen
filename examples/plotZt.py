# %% plot Zt.csv in the base directory 

from matplotlib import pyplot as plt
import numpy as np

X1 = np.genfromtxt('../X1.csv', delimiter=',')[:, :-1]
Zt = np.genfromtxt('../Zt.csv', delimiter=',')[:, :-1]
# covariance of Zt
print(np.cov(Zt.T))
# %%
plt.scatter(Zt[:,0], Zt[:,1], s=1)
plt.scatter(X1[:,0], X1[:,1], s=1)
plt.axis([-10,10,-10,10])
# %%
