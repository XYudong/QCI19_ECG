import tensorflow as tf
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras import layers, regularizers
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from numpy import save as np_save
import pickle
import math
import os

# environment configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # select to use which GPU

# print(tf.keras.__version__)
tf.keras.backend.set_image_data_format('channels_last')


def load_data_no_use(dataset='train'):
    if dataset not in ['train', 'test', 'val']:
        raise ValueError('Invalid dataset: ' + str(dataset))

    root = './data/ECG200/2D/'
    with open(root + 'ECG200_2D_' + dataset + '_aug02.pkl', 'rb') as infile:
        x, y = pickle.load(infile)
    return x, y


# def train_irv2(lr=1e-4, epochs=30):
#     x_train, y_train = load_data('train')
#     x_val, y_val = load_data('val')
#     x_test, y_test = load_data('test')
#     print('training set before resizing: ', x_train.shape)
#     print('validation set before resizing: ', x_val.shape)
#
#     batch_size = 16
#     train_dataset = get_dataset(x_train, y_train, batch_size)
#     val_dataset = get_dataset(x_val, y_val, batch_size)
#
#     # base pre-trained model
#     print('building model...')
#     base_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
#
#     # classifier
#     x = base_model.output       # a vector after GlobalAveragePooling2D(), shape: (?, 1536)
#     fc1 = layers.Dense(64, kernel_initializer=TruncatedNormal(),
#                        kernel_regularizer=regularizers.l2(0.1), name='clf_fc1')(x)
#     bn1 = layers.BatchNormalization()(fc1)
#     act1 = layers.Activation('relu', name='act_fc1')(bn1)
#     d1 = layers.Dropout(0.5)(act1)
#
#     # d2 = layers.Dropout(0.3)(act1)
#     fc2 = layers.Dense(2, kernel_initializer=TruncatedNormal(),
#                        kernel_regularizer=regularizers.l2(0.1), name='clf_fc2')(d1)
#     predictions = layers.Activation('sigmoid', name='prediction')(fc2)
#
#     model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
#
#     for layer in base_model.layers:
#         layer.trainable = False
#     model.get_layer('conv_7b').trainable = True
#
#     # compile
#     adam = tf.keras.optimizers.Adam(lr=lr)
#     model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # callbacks
#     checkpointer = ModelCheckpoint('./weight/irv2_ECG200_{val_acc:.2f}.h5',
#                                    monitor='val_loss',
#                                    save_best_only=True)
#     # reduce_lr = LearningRateScheduler(lr_scheduler)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                                   patience=3,
#                                   min_lr=1e-6)
#     tensorboard = TensorBoard(log_dir='./log/ECG200/01',
#                               write_graph=False,
#                               batch_size=batch_size)
#     print('start training...')
#     histoty = model.fit(train_dataset,
#                         steps_per_epoch=math.ceil(len(x_train)/batch_size),
#                         epochs=epochs,
#                         validation_data=val_dataset,
#                         validation_steps=math.ceil(len(x_val)/batch_size),
#                         callbacks=[checkpointer, reduce_lr, tensorboard],
#                         verbose=2)
#
#     # # Testing
#     [loss, acc] = model.evaluate(x_test, y_test, batch_size=len(y_test))
#     print('TEST loss: ', loss,)
#     print('TEST accuracy: ', acc)
#     return histoty


# def extract_feature(model_name, dataset_name='train'):
#     """extract feature vectors from the GlobalAveragePooling2D layer"""
#     if dataset_name not in ['train', 'test', 'val']:
#         raise ValueError('invalid dataset: ' + dataset_name)
#     x, y = load_data(dataset_name)
#     num_data = x.shape[0]
#     batch_size = 16
#
#     x = _parse_function(x)
#     print(dataset_name + ' set: ', x.shape)     # tensor
#     dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size).prefetch(1)
#
#     print('loading the model...')
#     model = load_model('./weight/' + model_name, compile=False)
#     model = tf.keras.Model(inputs=model.input,
#                            outputs=model.get_layer(name='avg_pool').output)
#
#     conv7b_avg_features = model.predict(dataset, verbose=1, steps=5)
#     # TODO: try manual generator for predict_generator
#
#     print('conv7b_avg_features shape: ', conv7b_avg_features.shape)
#     outfile = './fea_vector/' + model_name[0:-3] + dataset_name
#     # np_save(outfile, conv7b_avg_features)
#
#     return


if __name__ == '__main__':
    print('here')
    # hists = train_irv2(lr=1e-4, epochs=50)
    # get 0.6875 val_acc at most, which is not good
    # extract_feature('irv2_ECG200_0.66.h5', 'val')


