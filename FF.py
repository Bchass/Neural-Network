## Following guide: https://hackernoon.com/building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_sqaured_error
from tqdm import tqdm_notebook

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs

from collections import OrderedDict

cmaps = OrderedDict()

# Color map
cmaps['Qualitative (2)'] = [
    'Set1'
]

# Generate observations, 4 labels - multi class
data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
print(data.shape, labels.shape)

#visual for the data
plt.scatter(data[:,0], data[:,1], c=labels, cmap=cmaps)
plt.show()

#multi-class to binary
labels_orig = labels
labels = np.mod(labels_orig, 2)
plt.scatter(data[:0], data[:,1], c=labels,cmap=cmaps)
plt.show() 

#split binary data
X_train, X_val, Y_val = train_test_split(data, labels, strtify=labels, random_state=0)
print(X_train.shape, X_val.shape)