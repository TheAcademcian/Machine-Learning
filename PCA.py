# PCA 2D to 1D conversion
import numpy as np 
import matplotlib.pyplot as plt
x = np.array([[ 1, 2, 2, 30],
[ 2, 3, 2, 20],
[ 3, 2, 3, 25],
[ 8, 9, 2, 10],
[12, 7, 9, 5],
[11, 9, 4, 6],
[10, 9, 9, 8]])

n = np.shape(x)[0]
d = np.shape(x)[1]

print("The training data: \n", x)
print("The number of trainig data present is: ", n)
print("The number of features present in x is: ", d)

# performing preprocessing part 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x = sc.fit_transform(x) 
print(np.round(x,3))

# =============================================================================
# fig, ax = plt.subplots()
# ax.scatter(x[:,0], x[:,1], color='gray', marker='*')
# ax.set_xlabel('w1')
# ax.set_ylabel('w2')
# plt.show()
# =============================================================================

# Apply PCA
from sklearn.decomposition import PCA 
pca = PCA(n_components = 2) 
Z = pca.fit_transform(x) 
print(np.round(Z,3))

