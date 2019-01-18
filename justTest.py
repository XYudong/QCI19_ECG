import numpy as np
import tensorflow as tf
import pickle
import math


def load_data(dataset='train'):
    if dataset not in ['train', 'test', 'val']:
        raise ValueError('Invalid dataset: ' + str(dataset))

    root = './data/ECG200/2D/'
    with open(root + 'ECG200_2D_' + dataset + '.pkl', 'rb') as infile:
        x, y = pickle.load(infile)
    return x, y


# x_val, y_val = load_data('test')
# print(x_val.shape)
# print(y_val.shape)

aa = [[1,2,3], [2,3,4], [5,6,7]]
print([aa[i][0] for i in range(3)])

