# Cosine similariry
import numpy as np 

x = np.array([9,6,7,8])
y = np.array([1,1,3,5])
cosXY = sum(x*y)/(np.sqrt(np.sum(np.square(x)))*np.sqrt(np.sum(np.square(y))))
print("The data X:",x)
print("The data Y:",y)

print("The cosine similarity: ",np.round(cosXY,4))
inDegree = np.degrees(np.arccos(cosXY))
print("The degree: ",np.round(inDegree,4))

# The extended Jaccard coefficient 
x = np.array([1,0,1,1,1])
y = np.array([1,1,1,1,1])
EJC = sum(x*y)/(np.sum(x)+np.sum(y) - sum(x*y))
print("\nThe data X:",x)
print("The data Y:",y)

print("The extended Jaccard coefficient : ",np.round(EJC,4))
