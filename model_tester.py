import tensorflow as tf
from data_utils import get_data, get_dataset, img_standardization, _parse_function
import math


def evaluate_model(weight_name):
    batch_size = 16
    x_train, y_train = get_data(aug=True, name='train')
    x_test, y_test = get_data(aug=False, name='test')
    num_data = len(x_test)
    [x_test] = img_standardization(x_train, x_test)
    x_test = _parse_function(x_test, im_size=224)
    dataset_test = get_dataset(x_test, y_test, batch_size, resize=False)

    model = tf.keras.models.load_model('./weight/' + weight_name, compile=False)
    adam = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    [loss, acc] = model.evaluate(dataset_test, steps=math.ceil(num_data / batch_size))

    print('TEST loss: ', loss)
    print('TEST acc: ', acc)

    return


if __name__ == '__main__':
    evaluate_model('vgg16_ECG200_03.h5')
