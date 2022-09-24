# DBSCAN 
import numpy as np 
from sklearn.cluster import DBSCAN 

# Input data 
X = np.array([[1,2],[1,3],[1,4],[2,4],[2,5],[3,5],
              [3,6],[4,6],[5,5],[5,6],[6,4],[6,5],
              [6,6],[7,4],[7,5],[7,6],[4,2],[4,3],
              [5,1],[5,2],[6,1],[6,2],[7,1],[7,2],
              [8,1],[8,2],[9,1],[9,2],[9,3],[9,4],
              [9,5],[10,1],[10,2],[10,3],[10,4],[10,5]])

db = DBSCAN(eps=2, min_samples=6).fit(X) 
print(db.labels_)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool) 

#print(core_samples_mask)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_ 
  
# Number of clusters in labels, ignoring noise if present. 
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
  
#print(labels) 
  
# Plot result 
import matplotlib.pyplot as plt 
  
# Black removed and is used for noise instead. 
unique_labels = set(labels) 
colors = ['y', 'b', 'g', 'r'] 
print(colors) 
for k, col in zip(unique_labels, colors): 
    if k == -1: 
        # Black used for noise. 
        col = 'k'
  
    class_member_mask = (labels == k) 
  
    xy = X[class_member_mask & core_samples_mask] 
    plt.plot(xy[:, 0], xy[:, 1], 'X', markerfacecolor=col, 
                                      markeredgecolor='k',  
                                      markersize=6) 
  
    xy = X[class_member_mask & ~core_samples_mask] 
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
                                      markeredgecolor='k', 
                                      markersize=6) 
  
plt.title('number of clusters: %d' %n_clusters_) 
plt.show() 