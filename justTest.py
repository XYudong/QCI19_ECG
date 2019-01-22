import numpy as np
# import tensorflow as tf
import pickle
import math


def load_data(dataset='train'):
    if dataset not in ['train', 'test', 'val']:
        raise ValueError('Invalid dataset: ' + str(dataset))

    root = './data/ECG200/2D/'
    with open(root + 'ECG200_2D_' + dataset + '.pkl', 'rb') as infile:
        x, y = pickle.load(infile)
    return x, y


fea = np.load('./fea_vector/irv2_ECG200_0.66.npy')
print(fea.shape)

