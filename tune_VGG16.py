from tune_inceptionResnetv2_copy import load_data, prepare_data, _parse_function
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import layers, regularizers
from tensorflow.python.keras.initializers import TruncatedNormal
# from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
# from numpy import save as np_save
# import pickle
import math
import os

# environment configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # select to use which GPU

# print(tf.keras.__version__)
tf.keras.backend.set_image_data_format('channels_last')


def new_vgg16():
    model = VGG16(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Fine-tuning: freeze some layers
    for layer in model.layers[0:-8]:
        layer.trainable = False

    # rebuild the model
    # pool5 = Flatten(name='flatten')(model.outputs)        # this doesn't work!!
    flat = layers.Flatten(name='flatten')(model.layers[-1].output)

    dense_1 = layers.Dense(128, name='dense_1',
                           kernel_initializer=TruncatedNormal,
                           kernel_regularizer=regularizers.l2(0.01))(flat)
    bn_1 = layers.BatchNormalization()(dense_1)  # normalize the inputs of nonlinear layer(activation layer)
    act_1 = layers.Activation('relu')(bn_1)
    d1 = layers.Dropout(0.5, name='drop1')(act_1)

    dense_2 = layers.Dense(2, name='dense_2',
                           kernel_initializer=TruncatedNormal,
                           kernel_regularizer=regularizers.l2(0.01))(d1)
    # bn_3 = BatchNormalization()(dense_3)
    # prediction = Activation("softmax", name="softmax")(bn_3)
    prediction = layers.Activation("sigmoid", name="sigmoid")(dense_2)  # for binary classification

    model = tf.keras.Model(inputs=model.inputs, outputs=prediction)

    return model


def train_vgg16(lr=1e-4, epochs=40):
    x_train, y_train = load_data('train')
    x_val, y_val = load_data('val')
    # x_test, y_test = load_data('test')
    num_data = x_train.shape[0]
    print('training set before resizing: ', x_train.shape)
    print('validation set before resizing: ', x_val.shape)

    # parse numpy arrays into resized tensors
    x_train = _parse_function(x_train, im_size=224)
    x_val = _parse_function(x_val, im_size=224)

    batch_size = 16
    dataset_train = prepare_data(x_train, y_train, batch_size, resize=False)
    dataset_val = prepare_data(x_val, y_val, batch_size, resize=False)

    # build model
    print('building model...')
    model = new_vgg16()

    # compile
    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    checkpointer = ModelCheckpoint('./weight/vgg16_ECG200_{val_acc:.2f}.h5',
                                   monitor='val_loss',
                                   save_best_only=True)
    # reduce_lr = LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
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
    # # Testing
    # [loss, acc] = model.evaluate(x_test, y_test, batch_size=len(y_test))
    # print('TEST loss: ', loss, )
    # print('TEST accuracy: ', acc)
    return histoty


hist = train_vgg16(lr=1e-4, epochs=40)
