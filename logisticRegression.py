#The logistic regression using gradient descent
import matplotlib.pyplot as plt 
import numpy as np 


# Training data
x = np.array([
[1, 1,  1,  1,  1,  1,  1],
[1,	3,	16,	28,	9,	10,	42]])

# To know number of tuples and features
n = np.shape(x)[1]
d = (np.shape(x)[0]-1)
y = np.array([0,	0,	1,	1,	0,	0,	1])

print("The training data: \n", x)
print("The actual output: ",y)
print("The number of trainig data present is: ", n)
print("The number of features present in x is: ", d)

# Scatter plot of the training data
plt.scatter(x[1,:], y) 
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
plt.title('Training data') 
plt.show()

# Initialization of the parameters
theta = np.zeros((d+1,1)) # Let the initial value of theta
temp = np.zeros((len(x),1))
alpha = 0.0001 # Learning rate
print("The learning rate is: ", alpha)
print("\n")

# Calculate initial cost value
fx = (1/(1+np.exp(-theta.transpose().dot(x))))
oldCost = (1/n)*np.sum(-y*np.log(fx) - (1-y)*np.log(1 - fx))
print("The initial cost is: ", oldCost)

# Start logistic regression
costPlot = []
iteration = []
it = 0

while True:
    it +=1
    temp = theta - (alpha*np.sum((fx - y)*x, axis = 1)).reshape(-1,1)
    theta = temp
    fx = (1/(1+np.exp(-theta.transpose().dot(x))))
    
    # The cost and the iterations are upadted
    newCost = (1/n)*np.sum(-y*np.log(fx) - (1-y)*np.log(1 - fx))
    costDiff = oldCost - newCost
    oldCost = newCost
    costPlot.append(newCost)
    iteration.append(it)
    
    if (costDiff<0.00000005):
        break;


# The output and display section
print("The final cost value is: ", newCost)
print("Total Number of iterations is: ", it)
print("The Theta is: \n", theta)

# plotting of the cost function  
plt.plot(iteration, costPlot) 
plt.xlabel('Iterations') 
plt.ylabel('Cost') 
plt.title('Cost function') 
plt.show()

# Identification of the class label for the new data
newData = np.array([[1, 1], [5, 20]])
yNew = np.round((1/(1+np.exp(-theta.transpose().dot(newData)))))
print("The new data: ", newData[1,:])
print("The class labels of new data: ", yNew)

