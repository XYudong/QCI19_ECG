from pyts.image import GASF, MTF, RecurrencePlots
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical
import numpy as np
from cv2 import resize as cv_resize


def get_dataset(x, y, batch_size, resize):
    assert x.shape[0] == y.shape[0], "x and y with different length"
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if resize:
        dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size).repeat().prefetch(1)
    return dataset


def _parse_function(tensors, labels=None, im_size=299):
    tensors_resized = tf.image.resize_images(tensors, (im_size, im_size))
    if labels is not None:
        return tensors_resized, labels
    else:
        return tensors_resized


def load_ECG(name='train', ECG='ECG200'):
    if name not in ['train', 'test', 'val']:
        raise ValueError('Invalid dataset: ' + str(name))
    root = './data/'
    if ECG == 'ECG200':
        if name == 'train':
            fname = '/ECG200_train.txt'
        elif name == 'val':
            fname = '/ECG200_val.txt'
        else:
            fname = '/ECG200_TEST.txt'

        x, y = data_label_split(root + ECG + fname)
    else:
        raise ValueError("Unknown ECG: " + ECG)
    return x, y


def data_label_split(filename):
    data = np.loadtxt(filename, delimiter=',')
    labels = data[:, 0]
    xs = data[:, 1:]
    return xs, labels


def white_noise_augmentation_new(x, y, sigma, times=10):
    # times = int(times)
    # augmentation of 1D data sequence
    if len(x.shape) == 1:
        x = np.reshape(x, (1, -1))

    mu = 0
    # rows = int((times-1)/2)
    rows = times-1
    noises1 = np.random.normal(mu, sigma, (int(x.shape[0] * rows), x.shape[1]))
    # noises2 = np.random.normal(mu, 3*sigma, (int(x.shape[0] * (rows+1)), x.shape[1]))

    x1 = np.repeat(x, rows, axis=0) + noises1
    # x2 = np.repeat(x, rows, axis=0) + noises1
    x = np.concatenate((x, x1), axis=0)

    y = np.repeat(y, times)
    print('after augmentation', x.shape, y.shape)
    return x, y


def white_noise_augmentation(x, y, times=3):
    # augmentation of 1D data
    mu, sigma = 0, 0.1
    x = np.repeat(x, 2, axis=0)
    y = np.repeat(y, 2, axis=0)
    for i in range(0, times):
        noise = np.random.normal(mu, sigma, x.shape)
        x1 = x + noise
        x = np.concatenate((x, x1), axis=0)
        y = np.concatenate((y, y), axis=0)
    print('after augmentation: ', x.shape, y.shape)
    return x, y


def transform_ECG(x, method):
    # transform ECG sequence(s) to binary image(s)
    if method == 'gasf':
        gasf = GASF(image_size=x.shape[1] // 2, overlapping=False, scale=-1)
        x = gasf.fit_transform(x)
        # print('applying GASF')
    elif method == 'mtf':
        mtf = MTF(image_size=x.shape[1], n_bins=4, quantiles='empirical', overlapping=False)
        x = mtf.fit_transform(x)
        # print('applying MTF')
    elif method == 'rp':
        rp = RecurrencePlots(dimension=1, epsilon='percentage_points', percentage=10)
        x = rp.fit_transform(x)
        # print('applying RP')
    else:
        raise ValueError("Invalid method: " + str(method))

    return x


def ECG2rgb(x, method):
    """transform ECG series into three-channel images"""
    num_data, ts_len = x.shape
    if method == 'comb':
        x_channels = []
        methods = ['rp', 'gasf', 'mtf']
        for method in methods:
            single_channel = transform_ECG(x, method)
            # print(method, single_channel.shape)
            x_channels.append(single_channel)

        x_rgb = []
        for i in range(num_data):
            x_resized = [cv_resize(x_channels[j][i], (ts_len, ts_len)) for j in range(3)]
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


def split_ECG(path='./data/ECG200/', filename='ECG200_TRAIN.txt'):
    """split ECG dataset into different different portions"""
    x_train = np.loadtxt(path + filename, delimiter=',')
    x_train, x_val = train_test_split(x_train, test_size=0.1, random_state=88)
    print(x_train.shape)
    print(x_val.shape)

    with open('./data/ECG200/ECG200_train.txt', 'wb+') as file:
        np.savetxt(file, x_train, fmt='%1.5f', delimiter=',')
    with open('./data/ECG200/ECG200_val.txt', 'wb+') as file:
        np.savetxt(file, x_val, fmt='%1.5f', delimiter=',')


def img_standardization(x_train, *x_in):
    """
    standardize elements in x_in per channel according to x_train
    :param x_train: each element is a three-channel array
    :param x_in:
    :return: a list
    """
    x_mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    x_std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
    return [(x - x_mean) / x_std for x in x_in]


def get_data(aug, name):
    """get ECG data in rgb image format"""
    method = 'comb'
    x, y = load_ECG(name)
    if aug:
        # x, y = white_noise_augmentation_new(x, y, sigma=0.128, times=10)
        x, y = white_noise_augmentation(x, y, times=3)

    print('transforming ECG to images...')
    x = ECG2rgb(x, method)
    y = transform_label(y)
    print('images shape: ', x.shape, y.shape)

    return x, y


def main():
    method = 'comb'

    x_train, y_train = load_ECG(name='train')
    x_val, y_val = load_ECG(name='val')
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.1,
                                                      random_state=88)

    # x_train, y_train = white_noise_augmentation(x_train, y_train, 10)
    # x_val, y_val = white_noise_augmentation(x_val, y_val, 10)

    # xval_1, xval_2 = separate_classes(x_val, y_val)
    # xtr_1, xtr_2 = separate_classes(x_train, y_train)
    # xval_1_aug, yy = white_noise_augmentation(xval_1[0], [1], 10)
    # # print(xval_1[0])
    # plot_signal(xval_1[0], xval_1_aug[0:3])
    #
    # # plot_signal(xval_1, xval_2, num=10)
    # # plot_signal(xtr_1, xtr_2, num=30)
    #
    # plt.show()

    # print('transforming ECG to images...')
    # x_train = ECG2rgb(x_train, method)
    # y_train = transform_label(y_train)
    # x_val = ECG2rgb(x_val, method)
    # y_val = transform_label(y_val)
    # x_test = ECG2rgb(x_test, method)
    # y_test = transform_label(y_test)
    #
    # print(x_train.shape, x_val.shape, x_test.shape)
    #
    # root = './data/ECG200/2D/'
    # with open(root + 'ECG200_2D_train_aug02.pkl', 'wb+') as outfile:
    #     pickle.dump([x_train, y_train], outfile)
    # with open(root + 'ECG200_2D_val_aug02.pkl', 'wb+') as outfile:
    #     pickle.dump([x_val, y_val], outfile)
    # with open(root + 'ECG200_2D_test_aug02.pkl', 'wb+') as outfile:
    #     pickle.dump([x_test, y_test], outfile)
    #
    # print('done')


if __name__ == '__main__':
    main()
    # split_ECG(filename='ECG200_TRAIN.txt')
