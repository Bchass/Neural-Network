# Neural-Network outputing training data

import numpy as np 

# sigmoid function
def sigmoid(x,dirv=False):
    if (dirv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
