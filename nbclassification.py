# Bayes classification
import numpy as np

X = np.array([
        [1,1,1], 
        [2,1,1], 
        [2,2,1], 
        [2,3,2], 
        [1,3,2], 
        [1,3,2]])

# Print the number of data and dimension 
n = len(X)
d = len(X[0])-1

print("The K-NN algorithm: \n")
print("The training data: \n", X)
print("Total number of data: ",n)
print("Total number of features: ",d)

print("\nThe classes and the prior probability")
unique_elements, counts_elements = np.unique(X[:,d], return_counts=True)
summaryClass = np.asarray((unique_elements, counts_elements))
nClass = len(summaryClass[0])
print("Number of classes: ", nClass)
priorProbability = summaryClass[1,:]
summaryClass = np.append(summaryClass, [priorProbability/n], axis = 0)
print("The classes, frequency and the prior probability: \n", summaryClass)

MainSummary = []
for i in range(d):
    a = X[:,i]
    print("Original array:")
    print(a)
    unique_elements, counts_elements = np.unique(a, return_counts=True)
    print("Frequency of unique values of the said array:")
    summary = np.asarray((unique_elements, counts_elements))
    nDistinct = len(summary[0])
    print(summary)
    print(nDistinct)
    for j in range(nClass):
        
    
