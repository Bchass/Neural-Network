# Neural-Network calculating errors

import numpy as np 

# alpha param 
alphas = [0.001,0.01,0.1,1,10,100,1000]

# sigmoid function
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# output of sigmoid to driv
def sigmoid_to_dirv(output):
    return output*(1-output)

# input data
x = np.array( [[0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1]])

# output data
y = np.array ([[0],
             [1],
             [1],
             [0]])

np.random.seed(1)

# initialize weights with the mean of 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# loop through to get a smaller error
for j in range(100000):

    # Feed forward through layers
    layer0 = x
    layer1 = sigmoid(np.dot(layer0,syn0))
    layer2 = sigmoid(np.dot(layer1,syn1))

    # cacl error
    layer2e = y - layer2

    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(layer2e))))

    layer2d = layer2e*sigmoid(layer2,dirv=True)

    layer1e = layer2d.dot(syn1.T)

    layer1d = layer1e * sigmoid(layer1,dirv=True)

    syn1 += layer1.T.dot(layer2d)
    syn0 += layer0.T.dot(layer1d)
 