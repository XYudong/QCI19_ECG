from data_utils import load_ECG, white_noise_augmentation, white_noise_augmentation_new
import numpy as np
import matplotlib.pyplot as plt


def separate_classes(x, y):
    """separate different classes into different arrays"""
    normal_class = x[[i for i in range(len(y)) if y[i] == 1]]
    abnormal_class = x[[i for i in range(len(y)) if y[i] == -1]]
    return normal_class, abnormal_class


def plot_signal(x1, x2, num=10):
    if len(x1.shape) == 1:
        x1 = np.vstack((x1, x1))
    idx1 = np.random.choice(len(x1), num if num < len(x1) else len(x1), replace=False)
    idx2 = np.random.choice(len(x2), num if num < len(x2) else len(x2), replace=False)
    x1_new = x1[idx1]
    x2_new = x2[idx2]

    f1 = plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.ylim(-3, 5)
    for signal in x1_new:
        plt.plot(signal)
    plt.title('Normal in ECG200')
    plt.text(8, 3, 'without augmentation')

    plt.subplot(2, 1, 2)
    plt.ylim(-3, 5)
    for signal in x2_new:
        plt.plot(signal)
    plt.title('Abnormal in ECG200')
    # plt.text(8, 3, 'with augmentation')


def test_augmentation():
    x, y = load_ECG('train')
    print('x shape: ', x.shape)
    x1, y1 = white_noise_augmentation(x, y, times=3)
    x2, y2 = white_noise_augmentation_new(x, y, sigma=0.1)
    x3, y3 = white_noise_augmentation_new(x, y, sigma=0.2)
    x4, y4 = white_noise_augmentation_new(x, y, sigma=0.128)

    print('x1 shape: ', x1.shape)
    print('x2 shape: ', x2.shape)
    print('x3 shape: ', x3.shape)
    print('x4 shape: ', x4.shape)

    x_std = np.std(x)
    x1_std = np.std(x1)
    x2_std = np.std(x2)
    x3_std = np.std(x3)
    x4_std = np.std(x4)

    print('\nwithout augmentation, std: ', x_std)
    print('original aug w/ sigma 0.1, std: ', x1_std)
    print('\nnew aug w/ sigma 0.1, std: ', x2_std)
    print('new aug w/ sigma 0.2, std: ', x3_std)
    print('new aug w/ sigma 0.128, std: ', x4_std)

    return


if __name__ == '__main__':
    test_augmentation()
