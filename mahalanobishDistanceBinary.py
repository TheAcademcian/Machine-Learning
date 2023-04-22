# Mahalanobish distance

import numpy as np 
  
data = np.array([
[0,	1,	1],
[1,	1,	1],
[1,	0,	1],
[0,	0,	0],
[1,	1,	0]])

x = np.array([[1,0, 1]])
m = np.mean(data, axis = 0)
print("Mean: ", m)

xMm = x - m
print("The difference with mean xMm: ", xMm)

data = np.transpose(data)
covM = np.cov(data, bias = False)
invCoveM = np.linalg.inv(covM)

np.set_printoptions(suppress=True)
print("Covarinace matrix of data:\n", covM) 
print("Inv covarinace matrix of data:\n", invCoveM) 

tem1 = np.dot(xMm,invCoveM)
tem2 = np.dot(tem1, np.transpose(xMm))
print(tem1)
print(tem2) 
MD = np.sqrt(tem2)

print("The Mahalanobish distance: ", np.reshape(MD, -1))
