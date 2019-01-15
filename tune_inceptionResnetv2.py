import tensorflow as tf
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras import layers, regularizers
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
    x_test, y_test = load_data('test')
    print('training set: ', x_train.shape)

    batch_size = 32
    # zero-meaned inside preprocess_input function
    train_dataGen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_batches = train_dataGen.flow(x_train, y_train, batch_size=batch_size)

    # base pre-trained model
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')

    x = base_model.output       # a vector after GlobalAveragePooling2D()

    d1 = layers.Dropout(0.5)(x)
    fc1 = layers.Dense(1024, kernel_regularizer=regularizers.l2(0.01))(d1)
    bn1 = layers.BatchNormalization()(fc1)
    act1 = layers.Activation('relu', name='relu1')(bn1)

    d2 = layers.Dropout(0.4)(act1)
    fc2 = layers.Dense(2, kernel_regularizer=regularizers.l2(0.01))(d2)
    bn2 = layers.BatchNormalization()(fc2)
    predictions = layers.Activation('sigmoid', name='sigmoid')(bn2)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    checkpointer = ModelCheckpoint('./weight/irv2_ECG200.h5',
                                   monitor='val_acc',
                                   save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    print('start training...')
    histoty = model.fit_generator(train_batches,
                                  steps_per_epoch=len(x_train)//batch_size,
                                  epochs=epochs,
                                  validation_data=(x_val, y_val),
                                  callbacks=[checkpointer, reduce_lr],
                                  verbose=2,
                                  use_multiprocessing=True)

    # # Testing
    # [loss, acc] = model.evaluate(x_test, y_test, batch_size=len(y_test))
    # print('TEST loss: ', loss,)
    # print('TEST accuracy: ', acc)
    return histoty


def plot_acc_loss(hist):
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(hist.history['acc'], c='dodgerblue', linewidth=2)
    plt.plot(hist.history['val_acc'], c='r')
    xlim = plt.gca().get_xlim()
    plt.plot(xlim, [0.9, 0.9], '--', c='seagreen')
    plt.ylim(0.3, 1.0)
    plt.grid(True)
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'val acc ' + str(format(max(hist.history['val_acc']), '.3f'))], loc='lower right')

    # summarize history for loss
    # plt.figure(2, figsize=(8, 8))
    plt.subplot(212)
    plt.plot(hist.history['loss'], c='dodgerblue')
    plt.plot(hist.history['val_loss'], c='r')
    plt.grid(True)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper right')
    # figname2 = 'vgg16_ECG200_loss_latest'
    # plt.savefig(figname2)
    # figname = 'vgg16_ECG200_latest'
    # plt.savefig(figname)

    return True


hists = train_irv2(lr=1e-4, epochs=40)
for i, hist in enumerate(hists):
    plt.figure(i+1, figsize=(8, 10))
    plot_acc_loss(hist)

