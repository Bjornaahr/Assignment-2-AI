#%%
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_openml
from sklearn.datasets.samples_generator import make_blobs
from sklearn import datasets

#%%
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

#%%
def Kmean(X, K):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis = 0)

    # X coordinates of random centroids
    C_x = np.random.randint(0, np.max(X[:, 0]), size=K)
    # Y coordinates of random centroids
    C_y = np.random.randint(0, np.max(X[:, 1]), size=K)
    centroids = np.array(list(zip(C_x, C_y)), dtype=np.float32)

    centroidsNew = np.zeros(centroids.shape)    
    centroidsOld = np.zeros(centroids.shape)


    centroidsNew = deepcopy(centroids)

    clusters = np.zeros(len(X))

    
    error = dist(centroidsNew, centroidsOld, None)

    print(error)

    while error != 0:
        for i in range(len(X)):
            distances = dist(X[i], centroidsNew)
            cluster = np.nanargmin(distances)
            clusters[i] = cluster

        centroidsOld = deepcopy(centroidsNew)


        for i in range (K):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            centroidsNew[i] = np.nanmean(points, axis=0)
        error = dist(centroidsNew, centroidsOld, None)

        #if(np.isnan(error).any()):
         #   break

    return centroidsNew, clusters


#%%
maxClusters = range(2, 11)
#%%
X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=200)


#X, y = datasets.load_iris(return_X_y=True)
#X, y = fashion_mnist.load_data()



#%%
distortion =[]
for i in maxClusters:
    centroidsNew, clusters = Kmean(X, i)
    distortion.append(sum(np.min(cdist(X, centroidsNew, 'euclidean'), axis = 1)) / X.shape[0])




#%%
plt.plot(maxClusters, distortion)

#%%

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()

for i in range(0, len(centroidsNew)):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i % len(colors)])
ax.scatter(centroidsNew[:, 0], centroidsNew[:, 1], marker='*', s=200, c='orange')


#%%