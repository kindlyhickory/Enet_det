import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, concatenate, MaxPool2D, UpSampling2D,
    add, ReLU, BatchNormalization, Conv2DTranspose
)
from tensorflow.python.keras.models import Model


def ConvBNRelu(filters, kernel_size, strides=(1, 1), dilation=(1, 1)):
    def layer(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding='same',
                   use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    return layer


def bottleneck_reg(filters, rate, dilation=(1, 1)):
    def layer(x):
        in_filters = K.int_shape(x)[-1]
        conv1 = ConvBNRelu(int(in_filters / rate), (1, 1))(x)
        conv2 = ConvBNRelu(int(in_filters / rate), (3, 3), dilation=dilation)(conv1)
        conv3 = ConvBNRelu(filters, (1, 1))(conv2)

        if in_filters != filters:
            x = ConvBNRelu(filters, (1, 1))(x)
        out = add([conv3, x])
        return out

    return layer


def bottleneck_pool(filters, rate, pool):
    def layer(x):
        in_filters = K.int_shape(x)[-1]
        conv1 = ConvBNRelu(int(in_filters / rate), (2, 2), pool)(x)
        conv2 = ConvBNRelu(int(in_filters / rate), (3, 3))(conv1)
        conv3 = ConvBNRelu(filters, (1, 1))(conv2)

        if in_filters != filters:
            x = ConvBNRelu(filters, (1, 1))(x)

        x = MaxPool2D(pool, pool, padding='same')(x)
        out = add([conv3, x])
        return out

    return layer


def bottleneck_asymmetric(filters, rate):
    def layer(x):
        in_filters = K.int_shape(x)[-1]
        conv1 = ConvBNRelu(int(in_filters / rate), (1, 1))(x)
        conv2 = ConvBNRelu(int(in_filters / rate), (1, 5))(conv1)
        conv3 = ConvBNRelu(int(in_filters / rate), (5, 1))(conv2)
        conv4 = ConvBNRelu(filters, (1, 1))(conv3)

        out = add([conv4, x])
        return out

    return layer


def make_net(input_shape):
    img = Input(shape=input_shape, name='image')

    conv1 = ConvBNRelu(13, (3, 3), (2, 2))(img)
    pool1 = MaxPool2D((2, 2), (2, 2), padding='same')(img)
    cat1 = concatenate([conv1, pool1])

    bn_10 = bottleneck_pool(64, 4, (2, 2))(cat1)
    bn_11 = bottleneck_reg(64, 4)(bn_10)
    bn_12 = bottleneck_reg(64, 4)(bn_11)
    bn_13 = bottleneck_reg(64, 4)(bn_12)
    bn_14 = bottleneck_reg(64, 4)(bn_13)

    bn_20 = bottleneck_pool(128, 4, (2, 2))(bn_14)
    bn_21 = bottleneck_reg(128, 4)(bn_20)
    bn_22 = bottleneck_reg(128, 4, (2, 2))(bn_21)
    bn_23 = bottleneck_asymmetric(128, 4)(bn_22)
    bn_24 = bottleneck_reg(128, 4, (4, 4))(bn_23)
    bn_25 = bottleneck_reg(128, 4)(bn_24)
    bn_26 = bottleneck_reg(128, 4, (8, 8))(bn_25)
    bn_27 = bottleneck_asymmetric(128, 4)(bn_26)
    bn_28 = bottleneck_reg(128, 4, (16, 16))(bn_27)

    bn_30 = bottleneck_reg(128, 4)(bn_28)
    bn_31 = bottleneck_reg(128, 4, (2, 2))(bn_30)
    bn_32 = bottleneck_asymmetric(128, 4)(bn_31)
    bn_33 = bottleneck_reg(128, 4, (4, 4))(bn_32)
    bn_34 = bottleneck_reg(128, 4)(bn_33)
    bn_35 = bottleneck_reg(128, 4, (8, 8))(bn_34)
    bn_36 = bottleneck_asymmetric(128, 4)(bn_35)
    bn_37 = bottleneck_reg(128, 4, (16, 16))(bn_36)

    bn_UpSampling = ConvBNRelu(16, (3, 3))(bn_37)
    bn_UpSampling = Conv2DTranspose(16, (3, 3), (2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(bn_UpSampling)
    bn_UpSampling = BatchNormalization()(bn_UpSampling)
    bn_UpSampling = ReLU()(bn_UpSampling)

    bn_41 = bottleneck_reg(64, 4)(bn_UpSampling)
    bn_42 = bottleneck_reg(64, 4)(bn_41)

    out = Conv2D(3, (1, 1), use_bias=False, padding='same', activation='sigmoid')(bn_42)

    return Model(img, out)


if __name__ == '__main__':
    net = make_net((256, 256, 3))
    net.summary()
    test = np.zeros((1000, 256, 256, 3), np.float32)

    net.predict(test, batch_size=1, verbose=1)
    net.predict(test, batch_size=1, verbose=1)
