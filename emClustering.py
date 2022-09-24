#EM clustering algorithm
import numpy as np, numpy.random

k = 2
p = 2

X = np.array([
        [1,2], 
        [2,2], 
        [3,3],
        [10,10], 
        [11,10],
        [12,10]])

# Print the number of data and dimension 
n = len(X)
d = len(X[0])
addZeros = np.zeros((n, 1))
X = np.append(X, addZeros, axis=1)
print("The FCM algorithm: \n")
print("The training data: \n", X)
print("\nTotal number of data: ",n)
print("Total number of features: ",d)
print("Total number of Clusters: ",k)

#Initial guess of the mean
#meanC = numpy.random.randint(np.min(X), np.max(X), size=(1, k))
meanC = np.array([
        [3, 10]])
print("The initial mean: ", meanC)
sigma = 0.82

## Create an empty weights
weight = np.zeros((n,k))
#print("The initial weight: \n",weight)

for it in range(1):
    for i in range(n):
        #print("Ith Data",i)
        sumP =0
        for j in range(k):
            logL = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(np.square(X[i,0:d] - meanC[0,j])/(2*np.square(sigma))))
            #print("logL",logL)
            sumP +=0.5*logL
        #print("sumP",sumP)
        
        for j in range(k):
            logL = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-np.square(X[i,0:d] - meanC[0,j])/(2*np.square(sigma)))
            #print(0.5*logL/sumP)
            weight[i,j] = 0.5*logL/sumP
            #print(0.5*logL/sumP)
        #print(np.round(weight,2))

    sumPDF = []
    for j in range(k):
        sumPDF = np.append(sumPDF, np.sum(weight[:,j]))
    #print("sumPDF",sumPDF)
    
   # =============================================================================
    for j in range(k):
        #print("J",j)
        meanSum = 0
        for i in range(n):
        #print("I",i)
        #print(X[i])
        #print(sumPDF[j])
            meanSum += X[i]*weight[i,j]/sumPDF[j]
        print(meanSum)
    #print(meanSum[0])
        meanC[0,j] = meanSum[0]
print(meanC)
  
# =============================================================================
       
print("\nThe final weights: \n", np.round(weight,2))
for i in range(n):    
    cNumber = np.where(weight[i] == np.amax(weight[i]))
    X[i,d] = cNumber[0]
    
print("\nThe data with cluster number: \n", X)    
