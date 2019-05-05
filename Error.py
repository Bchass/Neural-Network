# Neural-Network calculating errors

import numpy as np

alphas = [0.001,0.01,0.1,1,10,100,1000]
hiddenSize = 32

# sigmoid function
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# output of sigmoid
def sigmoid_to_dirv(output):
    return output*(1-output)

# input data
x = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

# output data
y = np.array([[0],
             [1],
             [1],
             [0]])

for alpha in alphas:
    print ("\nTraining with alpah:" + str(alpha))
    np.random.seed(1)

    # initialize weights randomly
    weight0 = 2*np.random.random((3,hiddenSize)) - 1
    weight1 = 2*np.random.random((hiddenSize,1)) - 1

    for j in range(60000):

        # Feed forward prop
        layer0 = x
        layer1 = sigmoid(np.dot(layer0,weight0))
        layer2 = sigmoid(np.dot(layer1,weight1))

        # calc error
        layer2_error = layer2 - y

        if (j% 10000) == 0:
            print ("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer2_error))))

        # direction of target value
        layer2_delta = layer2_error*sigmoid_to_dirv(layer2)

        # layer1 value to layer2 value
        layer1_error = layer2_delta.dot(weight1.T)

        # direction of layer1 target value
        layer1_delta = layer1_error * sigmoid_to_dirv(layer1)

        weight1 -= alpha * (layer1.T.dot(layer2_delta))
        weight0 -= alpha * (layer0.T.dot(layer1_delta))