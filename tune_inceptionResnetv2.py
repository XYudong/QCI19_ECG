import tensorflow as tf
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras import layers, regularizers
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import pickle
import matplotlib.pyplot as plt
import os

# environment configuration
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # select to use which GPU

# print(tf.keras.__version__)
tf.keras.backend.set_image_data_format('channels_last')


def load_data(dataset='train'):
    if dataset not in ['train', 'test', 'val']:
        raise ValueError('Invalid dataset: ' + str(dataset))

    root = './data/ECG200/2D/'
    with open(root + 'ECG200_2D_' + dataset + '.pkl', 'rb') as infile:
        x, y = pickle.load(infile)
    return x, y


def train_irv2(lr=1e-4, epochs=30):
    x_train, y_train = load_data('train')
    x_val, y_val = load_data('val')
    # x_test, y_test = load_data('test')
    print('training set: ', x_train.shape)

    batch_size = 32
    train_dataGen = ImageDataGenerator(featurewise_center=True)
    train_dataGen.fit(x_train)
    train_batches = train_dataGen.flow(x_train, y_train, batch_size=batch_size)
    val_dataGen = ImageDataGenerator(featurewise_center=True)
    val_dataGen.fit(x_train)
    val_batch = val_dataGen.flow(x_val, y_val, batch_size=40)

    # base pre-trained model
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')

    x = base_model.output       # a vector after GlobalAveragePooling2D()
    d1 = layers.Dropout(0.1)(x)
    fc1 = layers.Dense(1024, kernel_initializer=TruncatedNormal(),
                       kernel_regularizer=regularizers.l2(0.01))(d1)
    bn1 = layers.BatchNormalization()(fc1)
    act1 = layers.Activation('relu', name='act_fc1')(bn1)

    # d2 = layers.Dropout(0.3)(act1)
    fc2 = layers.Dense(2, kernel_initializer=TruncatedNormal(),
                       kernel_regularizer=regularizers.l2(0.01))(act1)
    bn2 = layers.BatchNormalization()(fc2)
    predictions = layers.Activation('sigmoid', name='prediction')(bn2)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    checkpointer = ModelCheckpoint('./weight/irv2_ECG200.h5',
                                   monitor='val_acc',
                                   save_best_only=True)
    # reduce_lr = LearningRateScheduler(lr_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    tensorboard = TensorBoard(log_dir='./log/ECG200/01', write_images=True,
                              write_graph=False,
                              batch_size=batch_size)
    print('start training...')
    histoty = model.fit_generator(train_batches,
                                  steps_per_epoch=len(x_train)//batch_size,
                                  epochs=epochs,
                                  validation_data=val_batch,
                                  callbacks=[checkpointer, reduce_lr, tensorboard],
                                  verbose=2)

    # # Testing
    # [loss, acc] = model.evaluate(x_test, y_test, batch_size=len(y_test))
    # print('TEST loss: ', loss,)
    # print('TEST accuracy: ', acc)
    return histoty


hists = train_irv2(lr=1e-4, epochs=40)









