#%%
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.datasets import fetch_openml
from sklearn.datasets.samples_generator import make_blobs


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


K = 50
#%%
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=200)

m = X.shape[0]
n = X.shape[1]

mean = np.mean(X, axis=0)
std = np.std(X, axis = 0)
centroids = np.random.randn(K,n)*std+mean

#%%
centroidsOld = np.zeros(centroids.shape)
centroidsNew = deepcopy(centroids)

clusters = np.zeros(len(X))

#%%
error = dist(centroidsNew, centroidsOld, None)

while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], centroidsNew)
        cluster = np.argmin(distances)
        clusters[i] = cluster

    centroidsOld = deepcopy(centroidsNew)

    for i in range (K):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        centroidsNew[i] = np.mean(points, axis=0)
    error = dist(centroidsNew, centroidsOld, None)


#%%

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()

for i in range(K):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(centroidsNew[:, 0], centroidsNew[:, 1], marker='*', s=200, c='orange')


#%%