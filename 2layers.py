# Two layer network, with improving/optimizing errors/weights

import numpy as np

# sigmoid function
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# output of sigmoid to dirv
def sigmoid_to_dirv(output):
    return output*(1-output)

# input data
x = np.array([ [0,1],
               [0,1],
               [1,0],
               [1,0] ])

# output data
y = np.array ([[0,0,1,1]]).T

# random calculations
np.random.seed(1)

# weights randomly
weight1 = 2*np.random.random((2,1)) - 1

for iter in range(10000):

    # forward prop
    layer0 = x
    layer1 = sigmoid(np.dot(layer0,weight1))

    # error
    layer1e = layer1 - y

    # calc how much was missed in slope with sigmoid
    layer1d = layer1e * sigmoid_to_dirv(layer1)
    weight1_dirv = np.dot(layer0.T,layer1d)

    # update the weights
    weight1 -= weight1_dirv

print("Output data after training:")
print(layer1)