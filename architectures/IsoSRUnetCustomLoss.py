import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, GlobalAveragePooling3D # return the (batchsize, channel)
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D
from tensorflow.keras.layers import Permute, Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add as layer_add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from utils.loss_utils import L2TV, L2L2


# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')


def generate_iso_srunet_model_S3(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    # num_classes = gen_conf['num_classes']
    num_modalities = gen_conf['dataset_info'][dataset]['modalities']
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']
    shrink_dim = gen_conf['dataset_info'][dataset]['shrink_dim']
    sparse_scale = gen_conf['dataset_info'][dataset]['sparse_scale']
    thickness_factor = gen_conf['dataset_info'][dataset]['downsample_scale']

    downsize_factor = train_conf['downsize_factor']
    num_kernels = train_conf['num_kernels']
    num_filters = train_conf['num_filters']
    mapping_times = train_conf['mapping_times']
    num_levels = train_conf['num_levels']

    lr = train_conf['learning_rate']
    decay = train_conf['decay']

    input_shape = (num_modalities, ) + patch_shape
    output_shape = (num_modalities, ) + expected_output_shape

    assert dimension in [2, 3]

    model = __generate_iso_srunet_model_S3(
    dimension, num_modalities, input_shape, output_shape, activation, num_kernels=num_kernels, downsize_factor=downsize_factor, num_filters=num_filters, thickness_factor=thickness_factor, n_bb_mapping=mapping_times, num_levels=num_levels)

    # optimizer
    if optimizer == 'Adam' :
        optimizer = Adam(lr=0.0001, decay=decay)
    elif optimizer == 'SGD' :
        optimizer =  SGD(lr=0.0001, nesterov=True)

    # loss
    if loss == 'l2tv':
        print('Use L2-TV loss function.')
        p = train_conf['loss_params'][0]
        weight = train_conf['loss_params'][1]
        l2tv = L2TV(p = p, weight = weight)
        loss = l2tv.total_variation_loss
    elif loss == 'l2l2':
        weight = train_conf['loss_params']
        l2l2 = L2L2(weight=weight)
        loss = l2l2.l2_l2

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def __generate_iso_srunet_model_S3(
    dimension, num_modalities, input_shape, output_shape, activation, downsize_factor=2, num_kernels=3, num_filters=64, thickness_factor = 2, n_bb_mapping=2, num_levels = 4) :
    '''
    anisotropic down-sample
    32(16,16,16)-64(8,8,8)-128(4,4,4)-256(2,2,2)-256(2,2,2)-128(4,4,4)-64(8,8,8)-32(16,16,16)-16(32,32,32)
    :param dimension:
    :param num_modalities:
    :param input_shape:
    :param output_shape:
    :param activation:
    :param shuffling_dim:
    :param sparse_scale:
    :param downsize_factor:
    :param num_kernels:
    :param num_filters:
    :return:
    '''
    input = Input(shape=input_shape)

    mp_kernel_size = None

    if thickness_factor == 2:
        assert num_levels > 1, "'n_levels' should be larger than 1. "
        conv_stack = []
        temp_sparse_scale = [1, 1, 1]  # [2, 2, 2]
        start_idx = 1
        conv = get_conv_core_1(dimension, input, int(num_filters*2**start_idx/downsize_factor), num_kernel=num_kernels, trainable=False) # 32x16 # common path

        # converging path
        for idx in range(start_idx+1, num_levels, 1):
            ## partial contraction
            pool = get_max_pooling_layer_1(dimension, conv, mp_kernel_size) # 16 x 16
            # bb block
            conv = get_shuffling_operation_1(dimension, conv, n_bb_mapping, temp_sparse_scale, trainable=True) # 32 x 32 # right side skip connection, self containing deconvo
            conv_stack.append(conv)
            conv = get_conv_core_1(dimension, pool, int(num_filters*2**idx/downsize_factor), num_kernel=num_kernels,  trainable=False) # 16 x 16 # left side residual block 1

        up = conv
        # extraction path with con
        for idx in range(num_levels-1, start_idx, -1):
            conv = get_conv_core_1(dimension, up, int(num_filters*2**idx/downsize_factor), num_kernel=num_kernels, trainable=False) # left side residual block 2
            up = get_deconv_layer_1(dimension, conv, int(num_filters*2**(idx - 1)/downsize_factor), trainable=True)
            up = concatenate([up, conv_stack[idx-start_idx-1]], axis=1) 

        for idx in range(start_idx, 0, -1):
            conv = get_conv_core_1(dimension, up, int(num_filters*2**idx/downsize_factor), num_kernel=num_kernels, trainable=True)
            up = get_deconv_layer_1(dimension, conv, int(num_filters*2**(idx - 1)/downsize_factor), trainable=True)

        conv = get_conv_core_1(dimension, up, int(num_filters / downsize_factor), num_kernel=num_kernels, trainable=True)
        pred = get_conv_fc_1(dimension, conv, num_modalities, trainable=True)

    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])




def generate_iso_srunet_model(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    activation = train_conf['activation']
    dimension = train_conf['dimension']
    # num_classes = gen_conf['num_classes']
    num_modalities = gen_conf['dataset_info'][dataset]['modalities']
    expected_output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']
    shrink_dim = gen_conf['dataset_info'][dataset]['shrink_dim']
    sparse_scale = gen_conf['dataset_info'][dataset]['sparse_scale']
    thickness_factor = gen_conf['dataset_info'][dataset]['downsample_scale']

    downsize_factor = train_conf['downsize_factor']
    num_kernels = train_conf['num_kernels']
    num_filters = train_conf['num_filters']
    mapping_times = train_conf['mapping_times']
    num_levels = train_conf['num_levels']

    lr = train_conf['learning_rate']
    decay = train_conf['decay']

    input_shape = (num_modalities, ) + patch_shape
    output_shape = (num_modalities, ) + expected_output_shape

    assert dimension in [2, 3]

    model = __generate_iso_srunet_model(
    dimension, num_modalities, input_shape, output_shape, activation, num_kernels=num_kernels, downsize_factor=downsize_factor, num_filters=num_filters, thickness_factor=thickness_factor, n_bb_mapping=mapping_times, num_levels=num_levels)

    # optimizer
    if optimizer == 'Adam' :
        optimizer = Adam(lr=lr, decay=decay)
    elif optimizer == 'SGD' :
        optimizer =  SGD(lr=lr, nesterov=True)

    # loss
    if loss == 'l2tv':
        print('Use L2-TV loss function.')
        p = train_conf['loss_params'][0]
        weight = train_conf['loss_params'][1]
        l2tv = L2TV(p = p, weight = weight)
        loss = l2tv.total_variation_loss
    elif loss == 'l2l2':
        weight = train_conf['loss_params']
        l2l2 = L2L2(weight=weight)
        loss = l2l2.l2_l2

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def __generate_iso_srunet_model(
    dimension, num_modalities, input_shape, output_shape, activation, downsize_factor=2, num_kernels=3, num_filters=64, thickness_factor = 2, n_bb_mapping=2, num_levels = 4) :
    '''
    anisotropic down-sample
    32(16,16,16)-64(8,8,8)-128(4,4,4)-256(2,2,2)-256(2,2,2)-128(4,4,4)-64(8,8,8)-32(16,16,16)-16(32,32,32)
    :param dimension:
    :param num_modalities:
    :param input_shape:
    :param output_shape:
    :param activation:
    :param shuffling_dim:
    :param sparse_scale:
    :param downsize_factor:
    :param num_kernels:
    :param num_filters:
    :return:
    '''
    input = Input(shape=input_shape)

    mp_kernel_size = None

    if thickness_factor == 2:
        assert num_levels > 1, "'n_levels' should be larger than 1. "
        conv_stack = []
        temp_sparse_scale = [1, 1, 1]  # [2, 2, 2]
        start_idx = 1
        conv = get_conv_core(dimension, input, int(num_filters*2**start_idx/downsize_factor), num_kernel=num_kernels) # 32x16

        # converging path
        for idx in range(start_idx+1, num_levels, 1):
            ## partial contraction
            pool = get_max_pooling_layer(dimension, conv, mp_kernel_size) # 16 x 16
            # bb block
            conv = get_shuffling_operation(dimension, conv, n_bb_mapping, temp_sparse_scale) # 32 x 32
            conv_stack.append(conv)
            conv = get_conv_core(dimension, pool, int(num_filters*2**idx/downsize_factor), num_kernel=num_kernels) # 16 x 16

        up = conv
        # extraction path with con
        for idx in range(num_levels-1, start_idx, -1):
            conv = get_conv_core(dimension, up, int(num_filters*2**idx/downsize_factor), num_kernel=num_kernels)
            up = get_deconv_layer(dimension, conv, int(num_filters*2**(idx - 1)/downsize_factor))
            up = concatenate([up, conv_stack[idx-start_idx-1]], axis=1)

        for idx in range(start_idx, 0, -1):
            conv = get_conv_core(dimension, up, int(num_filters*2**idx/downsize_factor), num_kernel=num_kernels)
            up = get_deconv_layer(dimension, conv, int(num_filters*2**(idx - 1)/downsize_factor))

        conv = get_conv_core(dimension, up, int(num_filters / downsize_factor), num_kernel=num_kernels)
        pred = get_conv_fc(dimension, conv, num_modalities)

    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])



def get_conv_core(dimension, input, num_filters, num_kernel=3) :
    x = input
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    assert num_kernel>1, "num_kernel must be greater than 1."
    if dimension == 2 :
        input = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(input)
        for idx in range(num_kernel):
            x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization(axis=1)(x)
    else :
        input = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same')(input)
        for idx in range(num_kernel):
            x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization(axis=1)(x)
    x = layer_add([input, x]) # Skip connection
    x = Activation('relu')(x)
    x = BatchNormalization(axis=1)(x)

    return x

def get_max_pooling_layer(dimension, input, pool_size = None) :
    if pool_size is None:
        pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPool2D(pool_size=pool_size)(input)
    else :
        return MaxPool3D(pool_size=pool_size)(input)

def get_deconv_layer(dimension, input, num_filters, kernel_size = None, strides = None) :
    if kernel_size is None or strides is None:
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

# Shuffling operation:
def get_shuffling_operation(dimension, input, n_bb_mapping, sparse_scale, shrinking_scale = 2) :
    """
    This is the 3D extension of FSRCNN (Xiaoou Tang, CVPR 2016).
    e.g. 2[1^3] -> 2[3^3] * 3-4 times -> 16[1^3] -> deconv 16[1x1x8]
    :param dimension: dimensionality of input except channel
    :param input: the input patch
    :param n_bb_mapping: number of hidden block (conv+relu+BN) in FSRCNN
    :param sparse_scale: shuffling scale with respect to "shuffling_dim"
    :return: output patch
    """
    assert dimension in [2, 3], "The invalid dimensionality of input."
    shrinking_shape = input.shape.as_list()
    original_num_filters = shrinking_shape[1] # input/output # filter
    shrinking_num_filters = shrinking_shape[1] // shrinking_scale  # shrinking filters
    up_kernel_size = sparse_scale # up-sampling layer
    up_strides = up_kernel_size # up-sampling layer

    x = Conv3D(shrinking_num_filters, kernel_size=(1, 1, 1), padding='same')(input)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=1)(x)
    for idx in range(n_bb_mapping):
        x = Conv3D(shrinking_num_filters, kernel_size=(3, 3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
    x = Conv3D(original_num_filters, kernel_size=(1, 1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=1)(x)
    x = layer_add([input, x])
    return get_deconv_layer(dimension, x, num_filters=original_num_filters, kernel_size=up_kernel_size, strides=up_strides)


##################################################################################3
#stage 3


def get_conv_core_1(dimension, input, num_filters, num_kernel=3, trainable=True) :
    x = input
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)
    assert num_kernel>1, "num_kernel must be greater than 1."
    if dimension == 2 :
        input = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(input)
        for idx in range(num_kernel):
            x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
            x = Activation('relu')(x)
            x = BatchNormalization(axis=1)(x)
    else :
        input = Conv3D(num_filters, kernel_size=(1, 1, 1), padding='same', trainable=trainable)(input)
        for idx in range(num_kernel):
            x = Conv3D(num_filters, kernel_size=kernel_size, padding='same',trainable=trainable)(x)
            x = Activation('relu')(x)
            x = BatchNormalization(axis=1, trainable=trainable)(x)
    x = layer_add([input, x]) # Skip connection
    x = Activation('relu')(x)
    x = BatchNormalization(axis=1, trainable=trainable)(x)

    return x

def get_max_pooling_layer_1(dimension, input, pool_size = None) :
    if pool_size is None:
        pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPool2D(pool_size=pool_size)(input)
    else :
        return MaxPool3D(pool_size=pool_size)(input)

def get_deconv_layer_1(dimension, input, num_filters, kernel_size = None, strides = None, trainable=True) :
    if kernel_size is None or strides is None:
        strides = (2, 2) if dimension == 2 else (2, 2, 2)
        kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides, trainable=trainable)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides, trainable=trainable)(input)

def get_conv_fc_1(dimension, input, num_filters, trainable=True) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size, trainable=trainable)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size, trainable=trainable)(input)

    # return Activation('relu')(fc)
    return fc

def organise_output_1(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    # pred = Permute((2, 1))(pred)
    ## no activation for image processing case.
    if activation == 'null':
        return pred
    else:
        return Activation(activation)(pred)

# Shuffling operation:
def get_shuffling_operation_1(dimension, input, n_bb_mapping, sparse_scale, shrinking_scale = 2, trainable=True) :
    """
    This is the 3D extension of FSRCNN (Xiaoou Tang, CVPR 2016).
    e.g. 2[1^3] -> 2[3^3] * 3-4 times -> 16[1^3] -> deconv 16[1x1x8]
    :param dimension: dimensionality of input except channel
    :param input: the input patch
    :param n_bb_mapping: number of hidden block (conv+relu+BN) in FSRCNN
    :param sparse_scale: shuffling scale with respect to "shuffling_dim"
    :return: output patch
    """
    assert dimension in [2, 3], "The invalid dimensionality of input."
    shrinking_shape = input.shape.as_list()
    original_num_filters = shrinking_shape[1] # input/output # filter
    shrinking_num_filters = shrinking_shape[1] // shrinking_scale  # shrinking filters
    up_kernel_size = sparse_scale # up-sampling layer
    up_strides = up_kernel_size # up-sampling layer

    x = Conv3D(shrinking_num_filters, kernel_size=(1, 1, 1), padding='same', trainable=trainable)(input)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=1, trainable=trainable)(x)
    for idx in range(n_bb_mapping):
        x = Conv3D(shrinking_num_filters, kernel_size=(3, 3, 3), padding='same', trainable=trainable)(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1, trainable=trainable)(x)
    x = Conv3D(original_num_filters, kernel_size=(1, 1, 1), padding='same', trainable=trainable)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=1, trainable=trainable)(x)
    x = layer_add([input, x])
    return get_deconv_layer_1(dimension, x, num_filters=original_num_filters, kernel_size=up_kernel_size, strides=up_strides, trainable=trainable)
