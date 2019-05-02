# Neural-Network outputing training data

import numpy as np
import time 

start = time.time()

# sigmoid function
def sigmoid(x,dirv=False):
    if(dirv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input data
x = np.array ([  [0,0,1],
                 [0,1,1],
                 [1,0,1],
                 [1,1,1] ])

x1 = np.array ([ [1,1,1],
                 [1,0,1],
                 [0,1,1],
                 [0,0,1] ])

# output data
y = np.array([[0,0,1,1]]).T

y1 = np.array([[1,1,0,0]]).T

# seeds random numbers to make calculations
np.random.seed(1)

# initalize weights with a mean of 0
w1 = 2*np.random.random((3,1)) - 1
w2 = 2*np.random.random((3,1)) - 1

# how many times to loop through the given data
for iter in range (400000):

    # forward prop
    layer0 = x
    layerx = x1
    layer1 = sigmoid(np.dot(layer0,w1))
    layer2 = sigmoid(np.dot(layerx,w2))

    # calculate error
    layer1_e = y - layer1
    layer2_e = y - layer2

    # calculate error based off slope
    layer1_delta = layer1_e * sigmoid(layer1,True)
    layer2_delta = layer2_e * sigmoid(layer2,True)

    # update the weights
    w1 += np.dot(layer0.T,layer1_delta)
    w2 += np.dot(layerx.T,layer2_delta)

end = time.time()
elapsed = end - start

print("Output data after training:")
print("\n")
print(layer1)
print("\n")
print(layer2)
print("\n")
print(elapsed)