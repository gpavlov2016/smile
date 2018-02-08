import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Input, BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from wandb.wandb_keras import WandbKerasCallback
import wandb
import smiledataset

run = wandb.init()
config = run.config

config.epochs=10

# load data
train_X, train_y, test_X, test_y = smiledataset.load_data()

# convert classes to vector
num_classes = 2
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)

img_rows, img_cols = train_X.shape[1:]

# add additional dimension
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], test_X.shape[2], 1)
test_X = np.repeat(test_X, 3, axis=3)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
train_X = np.repeat(train_X, 3, axis=3)
print(train_X.shape)
print(train_y.shape)

#train_X /= 255.0
#test_X /= 255.0

#model = Sequential()
#model.add(Flatten(input_shape=(img_rows, img_cols,1)))
#model.add(Dense(num_classes, activation='softmax') )
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

shp = train_X.shape[1:]
model_vgg16_conv = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=train_X.shape[1:], pooling=None, classes=2)
#model_vgg16_conv = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=train_X.shape[1:], pooling=None, classes=2)
#model_vgg16_conv = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=shp, pooling=None, classes=2)
model_vgg16_conv.summary()

for layer in model_vgg16_conv.layers[:-6]:
    layer.trainable = False

input = Input(shape=(64, 64, 3), name = 'image_input')
bn1 = BatchNormalization()(input)

output_vgg16_conv = model_vgg16_conv(bn1)


do = 0.3
#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dropout(do)(x)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dropout(do)(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dropout(do)(x)
#x = Dense(128, activation='relu', name='fc4')(x)
#x = Dropout(do)(x)
x = Dense(2, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(input=input, output=x)
#model.compile(keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy')
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])

model.summary()

model.fit(train_X, train_y,
    batch_size = 128,
    epochs=config.epochs, verbose=1,
    validation_data=(test_X, test_y), callbacks=[WandbKerasCallback()])

model.save("smile.h5")
