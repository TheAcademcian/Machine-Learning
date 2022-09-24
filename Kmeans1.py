#K-Means algorithm
import numpy as np
from scipy.spatial import distance
k = 2

X = np.array([
        [1,1], 
        [2,1], 
        [2,2], 
        [5,3], 
        [6,3], 
        [5,4]])

# Print the number of data and dimension 
n = len(X)
d = len(X[0])
addZeros = np.zeros((n, 1))
X = np.append(X, addZeros, axis=1)
print("The K Means algorithm: \n")
print("The training data: \n", X)
print("Total number of data: ",n)
print("Total number of features: ",d)
print("Total number of Clusters: ",k)

# Random selection of initial cluster centers
dup = np.array([])
while 1:
    ranIndex = np.random.randint(low=1, high=n, size=k)
    u, c = np.unique(ranIndex, return_counts=True)
    dup = u[c > 1]
    if dup.size == 0:
        break
C = X[ranIndex]
print("\n The initial cluster centers: \n", C[:,0:d])
print("\n")

# Main iteration starts
for it in range(10): # Total number of iterations
    for i in range(n): # Iterate each data
        minDist = 9999999999
        for j in range(k): # Iterate each cluster center
            # Distance calculation from centers
            dist = distance.euclidean(C[j,0:d], X[i,0:d])
            if (dist<minDist):
                minDist = dist
                clusterNumber = j
            X[i,d] = clusterNumber
            C[j,d] = clusterNumber
 
    # Group the data to calculate the mean
    for j in range(k):
        result = np.where(X[:,d] == j)
        C[j] = np.reshape(np.mean(X[result,0:(d+1)], axis=1),(d+1))

# Calculate cost value
cost = 0
for i in range(n):
    tempC = C[C[:,d] == X[i,d]]
    #print(distance.euclidean(X[i,0:d],tempC[:,0:d]))
    cost += distance.euclidean(X[i,0:d],tempC[:,0:d])
cost = cost/n

print("The Final cluster centers: \n", C)
print("\n The data with cluster number: \n", X)
print("\n The cost is: ", np.round(cost,4))
# End of cost value calculation



   

    