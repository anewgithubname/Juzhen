# %% plot Zt.csv in the base directory 

from matplotlib import pyplot as plt
import numpy as np

# mu = np.genfromtxt('../mu_tr.csv', delimiter=',')[0, :-1]
# cov = np.genfromtxt('../cov_tr.csv', delimiter=',')[:, :-1]

# X1 = np.random.multivariate_normal(mu.squeeze(), cov, 60000)

# # write to the csv
# np.savetxt('../X0.csv', X1, delimiter=',')
X0 = np.genfromtxt('../X0.csv', delimiter=',')[:, :-1]
X1 = np.genfromtxt('../X1.csv', delimiter=',')[:, :-1]
Zt = np.genfromtxt('../Zt.csv', delimiter=',')[:, :-1]

# covariance of Zt
# print(np.cov(Zt.T))
# %%

# plot the first 16 28 by 28 images from Zt
for t in range(0,1001,100):
    Zt = np.genfromtxt('../Zt_'+str(t)+'.csv', delimiter=',')[:, :-1]
    for i in range(49):
        plt.subplot(7, 7, i+1)
        img = Zt[i, :].reshape(28, 28)
        # clamp the values to 0-1
        img = np.clip(img, 0, 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
    plt.savefig('../Zt_'+str(t)+'.png')

# plt.scatter(Zt[:,0], Zt[:,1], s=1)
# plt.scatter(X1[:,0], X1[:,1], s=1)
# plt.axis([-10,10,-10,10])
# %% convert testcase.png to csv

from PIL import Image
import numpy as np

img = Image.open('../testcase.png')
img = img.resize((28, 28))
# to grayscale
img = img.convert('L')
img = np.array(img)
img = 1 - img / 255.0
plt.imshow(img)
img = img.flatten().reshape(1, -1)
np.savetxt('../testcase.csv', img, delimiter=',')