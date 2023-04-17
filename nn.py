# Neural network using Keras
# install tensorflow and Keras in Anaconda prompt
#pip install tensorflow
#pip install keras

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataset = np.array([[.2, .3, 0], [.2, .1, 0], [.4, .1, 0], [.3, .3, 0],
                   [.4, .4, 0], [.8, .7, 1], [.7, .7, 1], [.8, .6, 1], [.9, .8, 1],
                   [.9, .7, 1]])
# split training data into input (X) and output (y) variables
X = dataset[:,0:2]
y = dataset[:,2]
print("Input and output\n", X, y)

# define the keras model
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(3, activation='relu'))
#model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=100, batch_size=4)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in np.array([1,4,7,8]):
	print('%s => Calculated: %d, Actual: %d' % (X[i].tolist(), predictions[i], y[i]))
