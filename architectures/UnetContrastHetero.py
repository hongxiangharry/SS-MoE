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
from keras.optimizers import Adam, SGD
# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')
def generate_hetero_unet_model(gen_conf, train_conf) :
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

    downsize_factor = train_conf['downsize_factor']
    num_kernels = train_conf['num_kernels']
    num_filters = train_conf['num_filters']

    lr = train_conf['learning_rate']
    decay = train_conf['decay']

    input_shape = (num_modalities, ) + patch_shape
    output_shape = (num_modalities, ) + expected_output_shape

    assert dimension in [2, 3]

    model = __generate_hetero_unet_model2(
        dimension, num_modalities, input_shape, output_shape, activation, shuffling_dim=shrink_dim, sparse_scale=sparse_scale, downsize_factor=downsize_factor, num_kernels=num_kernels, num_filters=num_filters)
    if optimizer == 'Adam' :
        optimizer = Adam(lr=lr, decay=decay)
    elif optimizer == 'SGD' :
        optimizer =  SGD(lr=lr, nesterov=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

def __generate_hetero_unet_model2(
    dimension, num_modalities, input_shape, output_shape, activation, shuffling_dim, sparse_scale, downsize_factor=2, num_kernels=3, num_filters=64) :
    '''
    anisotropic down-sample
    (32,32,8)-(16,16,8)-(8,8,8)-(4,4,4)-(2,2,2)-(2,2,2)-(4,4,4)-(8,8,8)-(16,16,16)-(32,32,32)
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
    shuffling_dim = np.array(shuffling_dim) # np serialize
    sparse_scale = np.array(sparse_scale)   # np serialize
    input = Input(shape=input_shape)

    conv1 = get_conv_core(dimension, input, int(num_filters/downsize_factor), num_kernel=num_kernels) ## c128

    temp_sparse_scale = sparse_scale
    pool1 = get_max_pooling_layer(dimension, conv1, (2, 2, 1))
    conv1 = get_shuffling_operation(dimension, conv1, shuffling_dim, temp_sparse_scale) ## c32
    conv2 = get_conv_core(dimension, pool1, int(num_filters*2/downsize_factor), num_kernel=num_kernels) ## c256

    temp_sparse_scale = sparse_scale / [1, 1, 2]
    pool2 = get_max_pooling_layer(dimension, conv2, (2, 2, 1))
    conv2 = get_shuffling_operation(dimension, conv2, shuffling_dim, temp_sparse_scale) ## c128
    conv3 = get_conv_core(dimension, pool2, int(num_filters*4/downsize_factor), num_kernel=num_kernels) ## c512

    temp_sparse_scale = sparse_scale / [1, 1, 4]
    pool3 = get_max_pooling_layer(dimension, conv3)
    conv3 = get_shuffling_operation(dimension, conv3, shuffling_dim, temp_sparse_scale) ## c512
    conv4 = get_conv_core(dimension, pool3, int(num_filters*8/downsize_factor), num_kernel=num_kernels) ## c1024

    pool4 = get_max_pooling_layer(dimension, conv4)
    conv4 = get_shuffling_operation(dimension, conv4, shuffling_dim, temp_sparse_scale) ## c1024
    conv5 = get_conv_core(dimension, pool4, int(num_filters*16/downsize_factor)) ## c2048

    # pool5 = get_max_pooling_layer(dimension, conv5)
    # conv5 = get_shuffling_operation(dimension, conv5, shuffling_dim, sparse_scale) ## c64
    # conv6 = get_conv_core(dimension, pool5, int(num_filters*32/downsize_factor)) ## c512

    reshape1 = get_shuffling_operation(dimension, conv5, shuffling_dim, temp_sparse_scale) ## c2048


    conv6 = get_conv_core(dimension, reshape1, int(num_filters*8/np.prod(temp_sparse_scale)/downsize_factor)) ## c1024
    up6 = get_deconv_layer(dimension, conv6, int(num_filters*8/np.prod(temp_sparse_scale)/downsize_factor)) ## c1024
    up6 = concatenate([up6, conv4], axis=1) ## c1024+1024

    conv7 = get_conv_core(dimension, up6, int(num_filters*8/np.prod(temp_sparse_scale)/downsize_factor), num_kernel=num_kernels)  ## c1024
    up7 = get_deconv_layer(dimension, conv7, int(num_filters*4/np.prod(temp_sparse_scale)/downsize_factor)) ## c512
    up7 = concatenate([up7, conv3], axis=1) ## c512+512

    temp_sparse_scale = sparse_scale/[1, 1, 2]
    conv8 = get_conv_core(dimension, up7, int(num_filters*4/np.prod(temp_sparse_scale)/downsize_factor), num_kernel=num_kernels) ## c256
    up8 = get_deconv_layer(dimension, conv8, int(num_filters*2/np.prod(temp_sparse_scale)/downsize_factor)) ## c128
    up8 = concatenate([up8, conv2], axis=1) ## c128+128

    temp_sparse_scale = sparse_scale
    conv9 = get_conv_core(dimension, up8, int(num_filters*2/np.prod(temp_sparse_scale)/downsize_factor), num_kernel=num_kernels) ## c64
    up9 = get_deconv_layer(dimension, conv9, int(num_filters/np.prod(temp_sparse_scale)/downsize_factor)) ## c32
    up9 = concatenate([up9, conv1], axis=1) ## c32+32

    conv10 = get_conv_core(dimension, up9, int(num_filters/np.prod(temp_sparse_scale)/downsize_factor), num_kernel=num_kernels) ## c32
    pred = get_conv_fc(dimension, conv10, num_modalities) ## the FCNN layer.
    # pred = layer_add([input, pred]) # comment the skip connection for hetero. u-net
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])

def __generate_hetero_unet_model(
    dimension, num_modalities, input_shape, output_shape, activation, shuffling_dim, sparse_scale, downsize_factor=2, num_kernels=3, num_filters=64) :
    shuffling_dim = np.array(shuffling_dim) # np serialize
    sparse_scale = np.array(sparse_scale)   # np serialize
    input = Input(shape=input_shape)

    conv1 = get_conv_core(dimension, input, int(num_filters/downsize_factor), num_kernel=num_kernels) ## c32

    pool1 = get_max_pooling_layer(dimension, conv1)
    conv1 = get_shuffling_operation(dimension, conv1, shuffling_dim, sparse_scale) ## c8
    conv2 = get_conv_core(dimension, pool1, int(num_filters*2/downsize_factor), num_kernel=num_kernels) ## c64

    pool2 = get_max_pooling_layer(dimension, conv2)
    conv2 = get_shuffling_operation(dimension, conv2, shuffling_dim, sparse_scale) ## c16
    conv3 = get_conv_core(dimension, pool2, int(num_filters*4/downsize_factor), num_kernel=num_kernels) ## c128

    pool3 = get_max_pooling_layer(dimension, conv3)
    conv3 = get_shuffling_operation(dimension, conv3, shuffling_dim, sparse_scale) ## c32
    conv4 = get_conv_core(dimension, pool3, int(num_filters*8/downsize_factor), num_kernel=num_kernels) ## c256

    # pool4 = get_max_pooling_layer(dimension, conv4)
    # conv4 = get_shuffling_operation(dimension, conv4, shuffling_dim, sparse_scale) ## c64
    # conv5 = get_conv_core(dimension, pool4, int(1024/downsize_factor)) ## c512

    reshape1 = get_shuffling_operation(dimension, conv4, shuffling_dim, sparse_scale) ## c128

    # conv6 = get_conv_core(dimension, reshape1, int(1024/np.prod(sparse_scale)/downsize_factor)) ## c128
    # up6 = get_deconv_layer(dimension, conv6, int(512/np.prod(sparse_scale)/downsize_factor)) ## c64
    # up6 = concatenate([up6, conv4], axis=1) ## c64+64

    conv7 = get_conv_core(dimension, reshape1, int(num_filters*8/np.prod(sparse_scale)/downsize_factor), num_kernel=num_kernels)  ## c64
    up7 = get_deconv_layer(dimension, conv7, int(num_filters*4/np.prod(sparse_scale)/downsize_factor)) ## c32
    up7 = concatenate([up7, conv3], axis=1) ## c32+32

    conv8 = get_conv_core(dimension, up7, int(num_filters*4/np.prod(sparse_scale)/downsize_factor), num_kernel=num_kernels) ## c32
    up8 = get_deconv_layer(dimension, conv8, int(num_filters*2/np.prod(sparse_scale)/downsize_factor)) ## c16
    up8 = concatenate([up8, conv2], axis=1) ## c16+16

    conv9 = get_conv_core(dimension, up8, int(num_filters*2/np.prod(sparse_scale)/downsize_factor), num_kernel=num_kernels) ## c16
    up9 = get_deconv_layer(dimension, conv9, int(num_filters/np.prod(sparse_scale)/downsize_factor)) ## c8
    up9 = concatenate([up9, conv1], axis=1) ## c8+8

    conv10 = get_conv_core(dimension, up9, int(num_filters/np.prod(sparse_scale)/downsize_factor), num_kernel=num_kernels) ## c8
    pred = get_conv_fc(dimension, conv10, num_modalities) ## the FCNN layer.
    # pred = layer_add([input, pred]) # comment the skip connection for hetero. u-net
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

# Shuffling operation:
def get_shuffling_operation(dimension, input, shuffling_dim, sparse_scale) :
    """
    This is the 3D extension of periodic shuffling (equation (4) in Magic Pony CVPR 2016).
    :param dimension: dimensionality of input except channel
    :param input: the input patch
    :param shuffling_dim: dimension indices for shuffling
    :param sparse_scale: shuffling scale with respect to "shuffling_dim"
    :return: output patch
    """
    assert dimension in [2, 3], "The invalid dimensionality of input."
    output_shape = input.shape.as_list()
    # shuffling_dim = np.array(shuffling_dim) # np serialize
    # sparse_scale = np.array(sparse_scale) # np serialize
    # assert shuffling_dim.shape == sparse_scale.shape, "The shapes of shuffling_dim and sparse_scale don't match."
    output_shape[2:] = output_shape[2:]*sparse_scale
    output_shape[1] = output_shape[1]//sparse_scale[0]//sparse_scale[1]//sparse_scale[2] # channel goes first
    return Reshape(tuple(output_shape[1:]))(input) # reshape by C-language order

def forward_periodic_shuffle(patch, upsampling_rate=2) :
    """
    Function source: Ryu's ESPCN code.
    This is the 3D extension of periodic shuffling (equation (4) in Magic Pony CVPR 2016).
    Args:
        patch (numpy array): 3 or 4 dimensional array with the last dimension being the dt components
        upsampling_rate (int): upsampling rate

    Returns:
    """
    if patch.ndim == 3:
        if patch.shape[2] == (upsampling_rate ** 2):
            dim_i, dim_j, dim_filters = patch.shape
            # apply periodic shuffling:
            patch_ps = np.ndarray((dim_i * upsampling_rate,
                                   dim_j * upsampling_rate),
                                  dtype='float64')

            # Apply reverse shuffling (optional):
            shuffle_indices = [(i, j)
                               for j in xrange(upsampling_rate)
                               for i in xrange(upsampling_rate)]

            no_channels = dim_filters / (upsampling_rate ** 3)

            for (i, j) in shuffle_indices:
                patch_ps[i::upsampling_rate,
                j::upsampling_rate] \
                    = patch[:, :, np.mod(i, upsampling_rate) +
                                  np.mod(j, upsampling_rate) * upsampling_rate]

        else:
            dim_i, dim_j, dim_filters = patch.shape

            # apply periodic shuffling:
            patch_ps = np.ndarray((dim_i * upsampling_rate,
                                   dim_j * upsampling_rate,
                                   dim_filters / (upsampling_rate ** 2)), dtype='float64')

            shuffle_indices = [(i, j)
                               for j in xrange(upsampling_rate)
                               for i in xrange(upsampling_rate)]

            no_channels = dim_filters / (upsampling_rate ** 2)

            for c in xrange(dim_filters // (upsampling_rate ** 2)):
                for (i, j) in shuffle_indices:
                    patch_ps[i::upsampling_rate,
                    j::upsampling_rate,
                    c] = patch[:, :, np.mod(i, upsampling_rate) +
                                     np.mod(j, upsampling_rate) * upsampling_rate +
                                     c * (upsampling_rate ** 2)]

    elif patch.ndim == 4:
        dim_i, dim_j, dim_k, dim_filters = patch.shape

        # apply periodic shuffling:
        patch_ps = np.ndarray((dim_i * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_filters // (upsampling_rate ** 3)), dtype='float64')

        shuffle_indices = [(i, j, k) for k in xrange(upsampling_rate)
                           for j in xrange(upsampling_rate)
                           for i in xrange(upsampling_rate)]

        no_channels = dim_filters / (upsampling_rate ** 3)

        for c in xrange(dim_filters // (upsampling_rate ** 3)):
            for (i, j, k) in shuffle_indices:
                patch_ps[i::upsampling_rate, j::upsampling_rate, k::upsampling_rate, c] \
                    = patch[:, :, :, np.mod(i, upsampling_rate) +
                                     np.mod(j, upsampling_rate) * upsampling_rate +
                                     np.mod(k, upsampling_rate) * (upsampling_rate ** 2) +
                                     c * (upsampling_rate ** 3)]

    elif patch.ndim == 5:  # apply periodic shuffling to a batch of examples.
        batch_size, dim_i, dim_j, dim_k, dim_filters = patch.shape

        # Apply reverse shuffling (optional):
        shuffle_indices = [(i, j, k) for k in xrange(upsampling_rate)
                           for j in xrange(upsampling_rate)
                           for i in xrange(upsampling_rate)]

        patch_ps = np.ndarray((batch_size,
                               dim_i * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_j * upsampling_rate,
                               dim_filters // (upsampling_rate ** 3)), dtype='float64')

        no_channels = dim_filters // (upsampling_rate ** 3)

        for c in xrange(dim_filters // (upsampling_rate ** 3)):
            for (i, j, k) in shuffle_indices:
                patch_ps[:, i::upsampling_rate, j::upsampling_rate, k::upsampling_rate, c] \
                    = patch[:, :, :, :, np.mod(i, upsampling_rate) +
                                        np.mod(j, upsampling_rate) * upsampling_rate +
                                        np.mod(k, upsampling_rate) * (upsampling_rate ** 2) +
                                        c * (upsampling_rate ** 3)]
    return patch_ps
