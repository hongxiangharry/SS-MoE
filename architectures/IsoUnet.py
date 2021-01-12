import numpy as np

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate
from keras.layers.merge import add as layer_add
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')
def generate_iso_unet_model(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    # num_classes = gen_conf['num_classes']
    num_modalities = gen_conf['dataset_info'][dataset]['modalities']
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']
    downsize_factor = train_conf['downsize_factor']

    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']

    input_shape = (num_modalities, ) + patch_shape
    output_shape = (num_modalities, ) + expected_output_shape

    assert dimension in [2, 3], "Unexpected dimension of patch."

    model = __generate_iso_unet_model(
        dimension, num_modalities, input_shape, output_shape, activation, downsize_factor=downsize_factor)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def __generate_iso_unet_model(
    dimension, num_modalities, input_shape, output_shape, activation, downsize_factor=2):
    input = Input(shape=input_shape)

    conv1 = get_conv_core(dimension, input, int(64/downsize_factor)) ## c32
    pool1 = get_max_pooling_layer(dimension, conv1)

    conv2 = get_conv_core(dimension, pool1, int(128/downsize_factor)) ## c64
    pool2 = get_max_pooling_layer(dimension, conv2)

    conv3 = get_conv_core(dimension, pool2, int(256/downsize_factor)) ## c128
    pool3 = get_max_pooling_layer(dimension, conv3)

    conv4 = get_conv_core(dimension, pool3, int(512/downsize_factor)) ## c256
    pool4 = get_max_pooling_layer(dimension, conv4)

    conv4_ = get_conv_core(dimension, pool4, int(1024/downsize_factor)) ## c512
    up5_ = get_deconv_layer(dimension, conv4_, int(512/downsize_factor)) ## c256

    up5_ = concatenate([up5_, conv4], axis=1)

    conv5_ = get_conv_core(dimension, up5_, int(512/downsize_factor))  ## c256
    up5 = get_deconv_layer(dimension, conv5_, int(256/downsize_factor))  ## c128
    up5 = concatenate([up5, conv3], axis=1)

    conv5 = get_conv_core(dimension, up5, int(256/downsize_factor)) ## c128
    up6 = get_deconv_layer(dimension, conv5, int(128/downsize_factor)) ## c64
    up6 = concatenate([up6, conv2], axis=1)

    conv6 = get_conv_core(dimension, up6, int(128/downsize_factor)) ## c64
    up7 = get_deconv_layer(dimension, conv6, int(64/downsize_factor)) ## c32
    up7 = concatenate([up7, conv1], axis=1)

    ## Comment the FCNN layer.
    conv7 = get_conv_core(dimension, up7, int(64/downsize_factor)) ## c32

    pred = get_conv_fc(dimension, conv7, num_modalities)
    # pred = layer_add([input, pred])
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])


def get_conv_core(dimension, input, num_filters) :
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(input)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(input)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)

    return x

def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPooling2D(pool_size=pool_size)(input)
    else :
        return MaxPooling3D(pool_size=pool_size)(input)

def get_deconv_layer(dimension, input, num_filters) :
    strides = (2, 2) if dimension == 2 else (2, 2, 2)
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)

def get_conv_fc(dimension, input, num_filters) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size)(input)

    # return Activation('relu')(fc)
    return fc

def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    # pred = Permute((2, 1))(pred)
    ## no activation for image processing case.
    if activation == 'null':
        return pred
    else:
        return Activation(activation)(pred)

