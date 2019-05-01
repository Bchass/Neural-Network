# Neural-Network outputing training data

import numpy as np 

# sigmoid function
def sigmoid(x,dirv=False):
    if (dirv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input data
x = np.array ([ [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# output data
y = np.array ([[1,0,0,1]]).T
