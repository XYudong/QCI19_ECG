from pyts.image import GASF, MTF, RecurrencePlots
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical
import numpy as np
import pickle
import cv2


def load_ECG(dataset='ECG200'):
    # root = "../data/"
    # x_train, y_train = data_label_split(root+dataset+'/'+dataset+'/'+dataset+'_TRAIN.txt')
    # x_test, y_test = data_label_split(root+dataset+'/'+dataset+'/'+dataset+'_TEST.txt')
    root = 'data/'
    if dataset == 'ECG5000':
        fname_tr = 'ECG5000_class1_2_train.csv'
        fname_te = 'ECG5000_class1_2_test.csv'

        x, y = data_label_split(root + dataset + '/' + 'ECG5000_class1_2_all.csv')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=88)
    elif dataset == 'ECG200':
        fname_train = 'ECG200_TRAIN.txt'
        fname_test = 'ECG200_TEST.txt'

        x_train, y_train = data_label_split(root + dataset + '/' + fname_train)
        x_test, y_test = data_label_split(root + dataset + '/' + fname_test)
    else:
        raise ValueError("Unknown dataset: " + dataset)
    return x_train, y_train, x_test, y_test


def data_label_split(filename):
    data = np.loadtxt(filename, delimiter=',')
    labels = data[:, 0]
    xs = data[:, 1:]
    return xs, labels


def white_noise_augmentation(x, y, times=2):
    times = int(times)
    # augmentation of 1D data sequence
    mu, sigma = 0, 0.1

    noises = np.random.normal(mu, sigma, (int(x.shape[0]*(times-1)), x.shape[1]))
    x1 = np.repeat(x, times-1, axis=0) + noises
    x = np.concatenate((x, x1), axis=0)

    y = np.repeat(y, times)
    print(x.shape, y.shape)
    return x, y


def transform_ECG(x, method):
    # transform ECG sequence(s) to binary image(s)
    if method == 'gasf':
        gasf = GASF(image_size=x.shape[1] // 2, overlapping=False, scale=-1)
        x = gasf.fit_transform(x)
        print('applying GASF')
    elif method == 'mtf':
        mtf = MTF(image_size=x.shape[1] // 3, n_bins=4, quantiles='empirical', overlapping=False)
        x = mtf.fit_transform(x)
        print('applying MTF')
    elif method == 'rp':
        rp = RecurrencePlots(dimension=1, epsilon='percentage_points', percentage=10)
        x = rp.fit_transform(x)
        print('applying RP')
    else:
        raise ValueError("Invalid method: " + str(method))

    return x


def preprocess(x, method):
    """transform ECG series into images"""
    if method == 'comb':
        x_channels = []
        methods = ['rp', 'gasf', 'mtf']
        for method in methods:
            single_channel = transform_ECG(x, method)
            # print(method, single_channel.shape)
            x_channels.append(single_channel)

        num_data = len(x_channels[0])
        ts_len = x.shape[1]
        x_rgb = []
        for i in range(num_data):
            x_resized = [cv2.resize(x_channels[j][i], (ts_len, ts_len)) for j in range(3)]
            img = np.stack(x_resized, axis=2)
            x_rgb.append(img)
        x_rgb = np.array(x_rgb)
        return x_rgb


def transform_label(y):
    num_classes = len(np.unique(y))
    # # transform raw class vector to integers from 0 to num_classes
    y = (y - y.min()) / (y.max() - y.min()) * (num_classes - 1)
    # Converts a class vector (integers) to binary class matrix, because of the use of loss='categorical_crossentropy'.
    Y = to_categorical(y, num_classes)

    return Y


def main():
    method = 'comb'

    x_train, y_train, x_test, y_test = load_ECG('ECG200')
    x_train, y_train = white_noise_augmentation(x_train, y_train, 8)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=88)

    print('transforming ECG to images...')
    x_train = preprocess(x_train, method)
    y_train = transform_label(y_train)
    x_val = preprocess(x_val, method)
    y_val = transform_label(y_val)
    x_test = preprocess(x_test, method)
    y_test = transform_label(y_test)

    print(x_train.shape, x_val.shape, x_test.shape)

    root = './data/ECG200/2D/'
    with open(root + 'ECG200_2D_train.pkl', 'wb+') as outfile:
        pickle.dump([x_train, y_train], outfile)
    with open(root + 'ECG200_2D_val.pkl', 'wb+') as outfile:
        pickle.dump([x_val, y_val], outfile)
    with open(root + 'ECG200_2D_test.pkl', 'wb+') as outfile:
            pickle.dump([x_test, y_test], outfile)

    print('done')


if __name__ == '__main__':
    main()
