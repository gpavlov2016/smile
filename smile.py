import numpy as np
import keras
import keras.models
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Input, BatchNormalization, concatenate
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import wandb 
from wandb.wandb_keras import WandbKerasCallback
import smiledataset

run = wandb.init()
config = run.config

config.epochs = 50
config.dropout = 0.5
config.batch_size = 32
config.lr = 1e-4
config.num_classes = 2

# load data
train_X, train_y, test_X, test_y = smiledataset.load_data()

# convert classes to vector
train_y = np_utils.to_categorical(train_y, config.num_classes)
test_y = np_utils.to_categorical(test_y, config.num_classes)

img_rows, img_cols = train_X.shape[1:]

# add additional dimension
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
test_X = np.repeat(test_X, 3, axis=3)  #VGG19 expects input with 3 channels
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
train_X = np.repeat(train_X, 3, axis=3) #VGG19 expects input with 3 channels
print('train_X.shape: ', train_X.shape)
print('train_y.shape:', train_y.shape)
print('test_X.shape: ', test_X.shape)
print('test_y.shape:', test_y.shape)


shp = train_X.shape[1:]
model_vgg19_conv = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=shp, pooling=None, classes=2)
#model_inception_resnet_v2 = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=config.num_classes)
model_vgg19_conv.summary()

#for layer in model_vgg19_conv.layers[:-10]:
#    print('Freezing layer: ', layer.name)
#    layer.trainable = False

datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zca_whitening=True)

valgen = ImageDataGenerator(zca_whitening=True)


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_X)
#valgen.fit(test_X)

input = Input(shape=shp, name = 'image_input')
bn1 = BatchNormalization()(input)

output_vgg19_conv = model_vgg19_conv(bn1)
block1_pool = model_vgg19_conv.get_layer('block1_pool')(bn1)
block2_pool = model_vgg19_conv.get_layer('block2_pool')(bn1)
block3_pool = model_vgg19_conv.get_layer('block3_pool')(bn1)
block4_pool = model_vgg19_conv.get_layer('block4_pool')(bn1)

block1_pool = Flatten(name='flatten_block1_pool')(block1_pool)
block2_pool = Flatten(name='flatten_block2_pool')(block2_pool)
block3_pool = Flatten(name='flatten_block3_pool')(block3_pool)
block4_pool = Flatten(name='flatten_block4_pool')(block4_pool)


block3_pool = BatchNormalization()(block3_pool)
block4_pool = BatchNormalization()(block4_pool)

#Add the fully-connected layers 
x = Flatten(name='flatten_x')(output_vgg19_conv)
x = BatchNormalization()(x)
x = keras.layers.concatenate([x, block4_pool, block3_pool]) #skip connections
x = Dropout(config.dropout)(x)
#x = BatchNormalization()(x)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dropout(config.dropout)(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dropout(config.dropout)(x)
#x = Dense(512, activation='relu', name='fc4')(x)
#x = Dropout(config.dropout)(x)
x = Dense(2, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(input=input, output=x)
#model.compile(keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy')
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=config.lr), metrics=['accuracy'])
model.summary()

# checkpoint
#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, WandbKerasCallback()]


from shutil import copyfile
import os
import matplotlib.pyplot as plt


if os.path.isfile(filepath):
  print('Backing up weights file')
  copyfile(filepath, filepath + '.backup')
#  print('Loading weights from ', filepath)
#  model.load_weights(filepath)

model.fit_generator(datagen.flow(train_X, train_y, batch_size = config.batch_size),
                    steps_per_epoch = len(train_X) / config.batch_size,
                    epochs=config.epochs, verbose=1,
                    validation_data=datagen.flow(test_X, test_y), 
#                    validation_data=(test_X, test_y), 
                    callbacks=callbacks_list)

'''
model.fit(train_X, train_y,
          batch_size = config.batch_size,
          epochs=config.epochs, verbose=1,
          validation_data=(test_X, test_y), 
          callbacks=callbacks_list)
'''


model.save("smile.h5")
