#Hierarchical clustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

#Input Data [x,y]
X = np.array([
        [.40,.53], 
        [.22,.38], 
        [.35,.32], 
        [.26,.19], 
        [.08,.41], 
        [.45,.30]])


# Print the number of data and dimension 
n = len(X)
d = len(X[0])
print("The hierarchical clustering algorithm: \n")
print("The training data: \n", X)
print("Total number of data: ",n)
print("Total number of features: ",d)

lab = range(1, 7)
plt.scatter(X[:,0],X[:,1], label='True Position')

for lab, x, y in zip(lab, X[:, 0], X[:, 1]):
    plt.annotate(lab, xy = (x, y), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
plt.show()

linkedData = linkage(X, 'single') # Apply single link distance
#linkedData = linkage(X, 'complete') 
#linkedData = linkage(X, 'average') 
#linkedData = linkage(X, 'centroid') 
#linkedData = linkage(X, 'median') 

labelList = range(1, 7)

# Draw the dendrogram
dendrogram(linkedData, orientation='top', labels=labelList, distance_sort='descending',
            show_leaf_counts=True)
plt.show()

   

    