from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as PCA_sklearn
from pca import PCA
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm


X_train, y_train = load_digits(return_X_y=True)

## CUSTOM PCA
pca = PCA()
pca.fit(X_train)

x_lim_left = 0
x_lim_right = 0
y_lim_left = 0
y_lim_right = 0

fig = plt.figure()
for x, label in tqdm(zip(X_train, y_train)):
    x_pos = np.dot(pca.components[0].T, x)
    y_pos = np.dot(pca.components[1].T, x)
    x_lim_right = max(x_pos, x_lim_right)
    x_lim_left = min(x_pos, x_lim_left)
    y_lim_right = max(y_pos, y_lim_right)
    y_lim_left = min(y_pos, y_lim_left)
    plt.text(x_pos, y_pos, str(label))
plt.xlim([x_lim_left,x_lim_right])
plt.ylim([y_lim_left,y_lim_right])
plt.title("Custom PCA")
plt.show(block=False)


## SKLEARN PCA
pca = PCA_sklearn(n_components=3)
pca.fit(X_train)

fig = plt.figure()
x_lim = 5
y_lim = 5
for x, label in zip(X_train, y_train):
    x_pos = np.dot(pca.components_[0].T, x)
    y_pos = np.dot(pca.components_[1].T, x)
    z_pos = np.dot(pca.components_[2].T, x)
    x_lim = max(x_lim, abs(x_pos))
    y_lim = max(y_lim, abs(y_pos))
    plt.text(x_pos, y_pos, str(label))
plt.xlim([x_lim_left,x_lim_right])
plt.ylim([y_lim_left,y_lim_right])
plt.title("Sklearn PCA")
plt.show()

