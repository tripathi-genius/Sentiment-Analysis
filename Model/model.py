from keras import layers
from keras.regularizers import l2
from keras.models import Model, Sequential
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Input
from keras.layers import Input, MaxPooling2D, SeparableConv2D, Flatten

def img_model(inShape, numberOfClasses):
    inputImage = Input(inShape)
    transform = Conv2D(32, (3, 3), strides=(2, 2), use_bias = False) (inputImage)
    transform = BatchNormalization(name='b1_conv1_bn')(transform)
    transform = Activation('relu', name='b1_conv1_act')(transform)
    transform = Conv2D(64, (3, 3), use_bias=False)(transform)
    transform = BatchNormalization(name='b1_conv2_bn')(transform)
    transform = Activation('relu', name='b1_conv2_act')(transform)
    remaining = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False) (transform)
    remaining = BatchNormalization()(remaining)
    remaining = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False) (transform)
    remaining = BatchNormalization()(remaining)
    transform = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(transform)
    transform = BatchNormalization(name='b2_sepConv1_bn')(transform)
    transorm = Activation('relu', name='b2_sepConv1_act')(transform)
    transform = SeparableConvolution2D(128, (3, 3), padding='same', use_bias=False)(transform)
    transform = BatchNormalization(name='b2_sepConv2_bn')(transform)
    transform = MaxPooling2D((3,3), strides=(2, 2), padding='same')(transform)
    transform = layers.add([transform, remaining])

    remaining = Conv2D(256, (3, 3), padding='same', use_bias=False)(transform)
    remaining = BatchNormalization()(remaining)
    transform = Activation('relu', name='b3_sepConv1_act')(transform)
    transform = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(transform)
    transform = BatchNormalization(name='b3_sepConv1_bn')(transform)
    transform = Activation('relu', name='b3_sepConv2_act')(transform)
    transform = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(transform)
    transform = BatchNormalization(name='b3_sepConv2_bn')(transform)
    transform = MaxPooling2D((3, 3), strides=(2, 2), padding='same',) (transform)
    transform = layers.add([transform, remaining])

    transform = Conv2D(numberOfClasses, (3, 3), padding='same')(transform)
    transform = GlobalAveragePooling2D()(transform)
    out = Activation('softmax', name='predictions')(transform)
    model = Model(inputImage, out)
    return model
