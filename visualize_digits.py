from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as PCA_sklearn
from pca import PCA
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.datasets import mnist

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape((-1,768))

# print(y_train.shape)

X_train = load_digits()['data']
y_train = load_digits()['target']

# X_train = X_train[:800,:100]
# y_train = y_train[:800]

mean_ = np.mean(X_train, axis=0)

X_train = X_train - mean_

# print(np.std(np.abs(X_train)+1e-9,axis=0))

# X_train = X_train / 255.

COV = np.cov(X_train.T)

vals, vecs = np.linalg.eigh(COV)
print(vals.min(),vals.max())
valvec = list(zip(vals, vecs.T))

valvec.sort(key=lambda x:abs(x[0]), reverse=True)

# vec1 = valvec[0][1]
vec1 = valvec[0][1] #* np.sign(valvec[0][0])
vec2 = valvec[1][1] #* np.sign(valvec[1][0])
vec3 = valvec[2][1] #* np.sign(valvec[1][0])

print(vec1)

print(X_train.shape)

pca = PCA()
pca.fit(X_train)


x_lim_left = 0
x_lim_right = 0
y_lim_left = 0
y_lim_right = 0
z_lim_left = 0
z_lim_right = 0
fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
for x, label in tqdm(zip(X_train, y_train)):
    # ptin
    x_pos = np.dot(pca.components[0].T, x)
    y_pos = np.dot(pca.components[1].T, x)
    # z_pos = np.dot(vec3.T, x)
    x_lim_right = max(x_pos, x_lim_right)
    x_lim_left = min(x_pos, x_lim_left)
    y_lim_right = max(y_pos, y_lim_right)
    y_lim_left = min(y_pos, y_lim_left)
    # z_lim_right = max(z_pos, z_lim_right)
    # z_lim_left = min(z_pos, z_lim_left)
    plt.text(x_pos, y_pos, str(label))
    # ax.text(x_pos, y_pos, z_pos, str(label))
plt.xlim([x_lim_left,x_lim_right])
plt.ylim([y_lim_left,y_lim_right])
# ax.set_zlim([z_lim_left,z_lim_right])
# plt.show()
plt.show(block=False)
pca = PCA_sklearn(n_components=3)
pca.fit(X_train)

print(pca.components_[0] == vec1)
fig = plt.figure()
x_lim = 5
y_lim = 5
# z_lim_left = 0
# z_lim_right = 0
# ax = fig.add_subplot(projection='3d')
for x, label in zip(X_train, y_train):
    # ptin
    x_pos = np.dot(pca.components_[0].T, x)
    y_pos = np.dot(pca.components_[1].T, x)
    z_pos = np.dot(pca.components_[2].T, x)
    x_lim = max(x_lim, abs(x_pos))
    y_lim = max(y_lim, abs(y_pos))
    plt.text(x_pos, y_pos, str(label))
    # ax.text(x_pos, y_pos, z_pos, str(label))
plt.xlim([x_lim_left,x_lim_right])
plt.ylim([y_lim_left,y_lim_right])
# ax.set_zlim([z_lim_left,z_lim_right])
plt.show()

# img = (30*pca.components_[0]-0*pca.components_[1]).reshape(8,8)
# img -= img.min()
# img /= img.max()
# plt.imshow(img, cmap="gray")
# plt.show()