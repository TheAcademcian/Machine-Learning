# Python code of Kendall's tau distance
import numpy as np
def kenDistance(a,b):
    data = np.matrix([a,b])
    data = data.transpose()
    data = data[np.argsort(data.A[:, 0])]
    data = data.transpose()
    a = data[0,:];b = data[1,:]
    a =np.array(a).flatten()
    b =np.array(b).flatten()
    d = a.shape[0]
    print(d)
    sumT = 0
    for i in range(d-2):
        for j in range(i+1,d-1):
            if b[i]>b[j]:
                sumT +=1
    print(sumT)
    return sumT/(d*(d-1)/2)
        
x = np.array([2,3,1,5,4])
y = np.array([4,2,1,3,5])
kenDist = kenDistance(x,y)
print("The Kendall's tau distance: ", kenDist)