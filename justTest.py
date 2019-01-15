import numpy as np
import pickle

aa = np.array([[1,2,3], [4,5,6], [0,1,1]])
bb = np.array([[1,1,1], [2,2,2]])

# cc = aa + bb
a = 2 // 3


a = np.array([[1],[2],[3]]).reshape((1,3,1))
b = np.array([[2],[3],[4]]).reshape((1,3,1))
c = np.dstack((a,b))

dataset = 'train'
if dataset not in ['train', 'test', 'val']:
    print('not here')
else:
    print('here')

print('training set: ', tuple((2,3,4)))

