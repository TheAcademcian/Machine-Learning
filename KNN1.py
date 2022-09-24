#K-NN algorithm
import numpy as np
from scipy.spatial import distance
from scipy import stats

k = 5

X = np.array([
        [1,1,1,2,1], 
        [2,1,2,2,1], 
        [2,2,3,2,1], 
        [5,3,4,2,2], 
        [6,3,5,6,2], 
        [5,4,6,4,2],
        [10,20,10,20,3],
        [17,25,19,23,3],
        [19,30,27,16,3],
        [16,21,15, 18,3]])

# Print the number of data and dimension 
n = len(X)
d = len(X[0])-1

print("The K-NN algorithm: \n")
print("The training data: \n", X)
print("Total number of data: ",n)
print("Total number of features: ",d)
print("The value of k is: ",k)

#testD = np.array([1,3,3,1])
testD = np.array([15,16,10,15])


dist = np.array([])
print("The test data is: ", testD)
# Main iteration starts
for i in range(n): # Iterate each data
    dist = np.append(dist, distance.euclidean(testD, X[i,0:1]))

indexS = np.argsort(dist) # return the indices after sorting

# The class labels of k number of data 
label = X[indexS[0:k],d] # select the first k indices
print("\nThe class labels are: ",label)
classV = stats.mode(label)
print("The class label of the new data is: ", classV[0])
  

    