# Two layer network, with improving/optimizing errors/weights

import numpy as np

# sigmoid function
def sigmoid(x):
    output: 1/(1+np.exp(-x))
    return output


