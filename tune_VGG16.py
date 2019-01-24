from data_utils import get_data, get_dataset, _parse_function, img_standardization
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import layers, regularizers
from tensorflow.python.keras.initializers import TruncatedNormal
# from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import numpy as np
# import pickle
import math
import os

# environment configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # select to use which GPU

# print(tf.keras.__version__)
tf.keras.backend.set_image_data_format('channels_last')


def new_vgg16():
    model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Fine-tuning: freeze some layers
    for layer in model.layers[0:-8]:
        layer.trainable = False

    # rebuild the model
    flat = layers.Flatten(name='flatten')(model.layers[-1].output)

    fc_1 = layers.Dense(128, name='fc_1',
                        # kernel_initializer=TruncatedNormal(),
                        kernel_regularizer=regularizers.l2(0.01))(flat)
    bn_1 = layers.BatchNormalization()(fc_1)  # normalize the inputs of nonlinear layer(activation layer)
    act_1 = layers.Activation('relu')(bn_1)
    d_1 = layers.Dropout(0.5, name='drop1')(act_1)

    # fc_2 = layers.Dense(128, name='fc_2',
    #                     # kernel_initializer=TruncatedNormal(),
    #                     kernel_regularizer=regularizers.l2(0.01))(act_1)
    # bn_2 = layers.BatchNormalization()(fc_2)
    # act_2 = layers.Activation('relu')(bn_2)

    fc_3 = layers.Dense(2, name='fc_3',
                        kernel_regularizer=regularizers.l2(0.01))(d_1)
    # prediction = Activation("softmax", name="softmax")(bn_3)
    prediction = layers.Activation("sigmoid", name="sigmoid")(fc_3)  # for binary classification

    model = tf.keras.Model(inputs=model.inputs, outputs=prediction)

    return model


def train_vgg16(lr=1e-4, epochs=50):
    x_train, y_train = get_data(aug=True, name='train')
    x_val, y_val = get_data(aug=True, name='val')
    x_test, y_test = get_data(aug=False, name='test')

    num_data = x_train.shape[0]
    num_test = x_test.shape[0]
    print('training set before preprocessing: ', x_train.shape)
    print('validation set before preprocessing: ', x_val.shape)

    [x_train, x_val, x_test] = img_standardization(x_train, x_train, x_val, x_test)

    # parse numpy arrays into resized tensors
    x_train = _parse_function(x_train, im_size=224)
    x_val = _parse_function(x_val, im_size=224)
    x_test = _parse_function(x_test, im_size=224)

    batch_size = 16
    dataset_train = get_dataset(x_train, y_train, batch_size, resize=False)
    dataset_val = get_dataset(x_val, y_val, batch_size, resize=False)
    dataset_test = get_dataset(x_test, y_test, batch_size, resize=False)

    # build model
    print('building model...')
    model = new_vgg16()

    # compile
    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    checkpointer = ModelCheckpoint('./weight/vgg16_ECG200_{val_acc:.2f}.h5',
                                   monitor='val_loss',
                                   save_best_only=True,
                                   )
    # reduce_lr = LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=3,
                                  min_lr=1e-6)
    tensorboard = TensorBoard(log_dir='./log/ECG200/vgg16/',
                              write_graph=False,
                              batch_size=batch_size)
    print('start training...')
    histoty = model.fit(dataset_train,
                        steps_per_epoch=math.ceil(num_data / batch_size),
                        epochs=epochs,
                        validation_data=dataset_val,
                        validation_steps=math.ceil(num_data / batch_size),
                        callbacks=[checkpointer, reduce_lr, tensorboard],
                        verbose=2)
    # Testing
    # [loss, acc] = model.evaluate(x_test, y_test, batch_size=batch_size)
    [loss, acc] = model.evaluate(dataset_test, steps=math.ceil(num_test / batch_size))
    print('TEST loss: ', loss, )
    print('TEST accuracy: ', acc)
    return histoty


hist = train_vgg16(lr=1e-4, epochs=40)
