# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:45:36 2020

@author: P S BISHNU
"""

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
scaler = StandardScaler()
import numpy as np
X = np.array([[1,2],[1,3],[1,4],[2,4],[2,5],[3,5],
              [3,6],[4,6],[5,5],[5,6],[6,4],[6,5],
              [6,6],[7,4],[7,5],[7,6],[4,2],[4,3],
              [5,1],[5,2],[6,1],[6,2],[7,1],[7,2],
              [8,1],[8,2],[9,1],[9,2],[9,3],[9,4],
              [9,5],[10,1],[10,2],[10,3],[10,4],[10,5]])

X_scaled = scaler.fit_transform(X)
# cluster the data into five clusters
dbscan = DBSCAN(eps=1, min_samples = 4)
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")