# Two layer network, with improving/optimizing errors/weights

import numpy as np

# sigmoid function
def sigmoid(x):
    output: 1/(1+np.exp(-x))
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

