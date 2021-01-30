'''
    patching_utils.py
    Date: 2018-12-01
    Author: Hongxiang Lin
    Affiliation: CMIC, UCL, UK

    Description: utilities of patching functions for 3D MRI image

    1. padding+overlapping patching
    2. overlapping patching
    3. shuffle patching
'''

import numpy as np
from numpy.random import shuffle
from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
import matplotlib.patches as patches
import matplotlib.pyplot as plt




def overlap_patching_test(gen_conf,
                     train_test_conf,
                     input_data,
                     output_data = None,
                     trainTestFlag = 'train',
                     representative_modality = 0) :
    dataset = train_test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_test_conf['bg_discard_percentage'] ## ?
    dimension = train_test_conf['dimension'] ## output image size
    modalities = dataset_info['modalities']
    assert modalities > 0
    num_classes = gen_conf['num_classes'] ## ?
    patch_shape = train_test_conf['patch_shape'] ## input patch size (32, 32, 8)
    sparse_scale = dataset_info['sparse_scale'] ## [1, 1, 4]
    if output_data is not None and trainTestFlag == 'train':
        extraction_step = train_test_conf['extraction_step']  ## shifting step (16, 16, 4)
        output_extraction_step = (modalities,) + tuple(np.array(extraction_step)*sparse_scale) ## (1, 16, 16, 16)
        output_shape = train_test_conf['output_shape'] ## (32, 32, 32)
        ## todo: judge if label_selector is necessary
        output_nominal_shape = tuple(np.array(patch_shape)*sparse_scale)
        output_selector = determine_output_selector(dimension, output_nominal_shape, output_shape)
        output_patch_shape = (modalities,) + output_shape ## (1, 32, 32, 32)
        output_patch = np.zeros((0,) + output_patch_shape, dtype=np.float32) ## output patch size

        minimum_non_bg = bg_discard_percentage * np.prod(output_shape)  ## number of non-background voxels in input patch
    else:
        output_patch = None
        extraction_step = train_test_conf['extraction_step_test']  ## shifting step (32, 32, 8)

    input_patch_shape = (modalities,) + patch_shape ## (1, 32, 32, 8)
    data_extraction_step = (modalities,) + extraction_step ## (1, 16, 16, 4)

    input_patch = np.zeros((0, ) + input_patch_shape, dtype=np.float32)
    len_input_data = len(input_data) # the first size of dimensions
    for idx in range(len_input_data):
        ## padding the original input image to make sure all patches can be extracted.
        pad_size = ()
        for dim in range(dimension):
            pad_size += (patch_shape[dim] // 2,)
        input_length = len(input_patch)

        if output_data is not None and trainTestFlag == 'train':
            ## padding the original output image to make sure all patches can be extracted
            output_pad_size = ()
            for dim in range(dimension):
                output_pad_size += (output_shape[dim] // 2,)
            output_length = len(output_patch)
            ## output
            output_vol = pad_both_sides(dimension, output_data[0], output_pad_size)
            output_data = np.delete(output_data, 0, 0)
            output_tmp_train = extract_patches(dimension, output_vol, output_patch_shape, output_extraction_step)
            del output_vol

            output_tmp_train = output_tmp_train[output_selector]
            ## dimension: patches x modalities x spatial size, considering modalities in this case
            sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)
            ## find indices of patches which has minimum backgrounds (0: background).
            valid_idxs = np.where(np.sum(output_tmp_train[:, representative_modality] != 0, axis=sum_axis) >= minimum_non_bg)
            N = max( valid_idxs[0].shape )

            output_patch = np.vstack((output_patch, np.zeros((N,) + output_patch_shape, dtype=np.float32)))

            output_patch[input_length:] = output_tmp_train[valid_idxs] # input length = output length
            del output_tmp_train

            # input
            input_vol = pad_both_sides(dimension, input_data[0], pad_size)
            input_data = np.delete(input_data, 0, 0)
            input_tmp_train = extract_patches(dimension, input_vol, input_patch_shape, data_extraction_step)
            del input_vol

            input_patch = np.vstack((input_patch, np.zeros((N,) + input_patch_shape, dtype=np.float32)))

            input_patch[input_length:] = input_tmp_train[valid_idxs]
            del input_tmp_train
        elif output_data is None and trainTestFlag == 'test':
            # no padding, no output selector, no overlapping
            ## todo: judge if the patches are overlapped.
            # input
            # input_vol = input_data[idx] # non-padding version
            input_vol = pad_both_sides(dimension, input_data[0], pad_size) # padding version
            input_data = np.delete(input_data, 0, 0)
            input_tmp_train = extract_patches(dimension, input_vol, input_patch_shape, data_extraction_step)
            del input_vol

            input_patch = np.vstack((input_patch, input_tmp_train))
            del input_tmp_train
        # else:
        #     ## todo: push error information

    ## shuffle patches and pick a subset
    '''
    if output_data is not None and trainTestFlag == 'train':
        psr = train_test_conf['patch_sampling_rate']
        num_sampling_patch = int(np.ceil(input_patch.shape[0]*psr))
        smpl_patch_indices = np.arange(input_patch.shape[0])
        shuffle(smpl_patch_indices)
        smpl_patch_indices = smpl_patch_indices[:num_sampling_patch]

        input_patch = input_patch[smpl_patch_indices]
        output_patch = output_patch[smpl_patch_indices]
    '''

    return input_patch, output_patch



def overlap_patching(gen_conf,
                     train_test_conf,
                     input_data,
                     output_data = None,
                     trainTestFlag = 'train',
                     representative_modality = 0) :
    dataset = train_test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_test_conf['bg_discard_percentage'] ## ?
    dimension = train_test_conf['dimension'] ## output image size
    modalities = dataset_info['modalities']
    assert modalities > 0
    num_classes = gen_conf['num_classes'] ## ?
    patch_shape = train_test_conf['patch_shape'] ## input patch size (32, 32, 8)
    sparse_scale = dataset_info['sparse_scale'] ## [1, 1, 4]
    if output_data is not None and trainTestFlag == 'train':

        # extraction_step is the same in training and test (4, 4, 4)
        extraction_step = train_test_conf['extraction_step']  ## shifting step (16, 16, 4)
        output_extraction_step = (modalities,) + tuple(np.array(extraction_step)*sparse_scale) ## (1, 16, 16, 16)
        output_shape = train_test_conf['output_shape'] ## (32, 32, 32)
        ## todo: judge if label_selector is necessary
        output_nominal_shape = tuple(np.array(patch_shape)*sparse_scale)
        output_selector = determine_output_selector(dimension, output_nominal_shape, output_shape)
        output_patch_shape = (modalities,) + output_shape ## (1, 32, 32, 32)
        output_patch = np.zeros((0,) + output_patch_shape, dtype=np.float32) ## output patch size

        minimum_non_bg = bg_discard_percentage * np.prod(output_shape)  ## number of non-background voxels in input patch
    else:
        output_patch = None
        extraction_step = train_test_conf['extraction_step_test']  ## shifting step (32, 32, 8)

    input_patch_shape = (modalities,) + patch_shape ## (1, 32, 32, 8)
    data_extraction_step = (modalities,) + extraction_step ## (1, 16, 16, 4)

    input_patch = np.zeros((0, ) + input_patch_shape, dtype=np.float32)
    len_input_data = len(input_data) # the first size of dimensions
    for idx in range(len_input_data):
        ## padding the original input image to make sure all patches can be extracted.
        pad_size = ()
        for dim in range(dimension):
            pad_size += (patch_shape[dim] // 2,)
        input_length = len(input_patch)

        if output_data is not None and trainTestFlag == 'train':
            ## padding the original output image to make sure all patches can be extracted
            output_pad_size = ()
            for dim in range(dimension):
                output_pad_size += (output_shape[dim] // 2,)
            output_length = len(output_patch)
            ## output
            output_vol = pad_both_sides(dimension, output_data[0], output_pad_size)
            output_data = np.delete(output_data, 0, 0)
            output_tmp_train = extract_patches(dimension, output_vol, output_patch_shape, output_extraction_step)
            del output_vol

            output_tmp_train = output_tmp_train[output_selector]
            ## dimension: patches x modalities x spatial size, considering modalities in this case
            sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)
            ## find indices of patches which has minimum backgrounds (0: background).
            valid_idxs = np.where(np.sum(output_tmp_train[:, representative_modality] != 0, axis=sum_axis) >= minimum_non_bg)
            N = max( valid_idxs[0].shape )

            output_patch = np.vstack((output_patch, np.zeros((N,) + output_patch_shape, dtype=np.float32)))

            output_patch[input_length:] = output_tmp_train[valid_idxs] # input length = output length
            del output_tmp_train

            # input
            input_vol = pad_both_sides(dimension, input_data[0], pad_size)
            input_data = np.delete(input_data, 0, 0)
            input_tmp_train = extract_patches(dimension, input_vol, input_patch_shape, data_extraction_step)
            del input_vol

            input_patch = np.vstack((input_patch, np.zeros((N,) + input_patch_shape, dtype=np.float32)))

            input_patch[input_length:] = input_tmp_train[valid_idxs]
            del input_tmp_train
        elif output_data is None and trainTestFlag == 'test':
            # no padding, no output selector, no overlapping
            ## todo: judge if the patches are overlapped.
            # input
            # input_vol = input_data[idx] # non-padding version
            input_vol = pad_both_sides(dimension, input_data[0], pad_size) # padding version
            input_data = np.delete(input_data, 0, 0)
            input_tmp_train = extract_patches(dimension, input_vol, input_patch_shape, data_extraction_step)
            del input_vol

            input_patch = np.vstack((input_patch, input_tmp_train))
            del input_tmp_train
        # else:
        #     ## todo: push error information

    ## shuffle patches and pick a subset
    '''
    # 20210126
    if output_data is not None and trainTestFlag == 'train':
        psr = train_test_conf['patch_sampling_rate']
        num_sampling_patch = int(np.ceil(input_patch.shape[0]*psr))
        smpl_patch_indices = np.arange(input_patch.shape[0])
        shuffle(smpl_patch_indices)
        smpl_patch_indices = smpl_patch_indices[:num_sampling_patch]

        input_patch = input_patch[smpl_patch_indices]
        output_patch = output_patch[smpl_patch_indices]
    '''

    return input_patch, output_patch

'''
def overlap_patching(gen_conf,
                     train_test_conf,
                     input_data,
                     output_data = None,
                     trainTestFlag = 'train',
                     representative_modality = 0) :
    dataset = train_test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    bg_discard_percentage = train_test_conf['bg_discard_percentage'] ## ?
    dimension = train_test_conf['dimension'] ## output image size
    modalities = dataset_info['modalities']
    assert modalities > 0
    num_classes = gen_conf['num_classes'] ## ?
    patch_shape = train_test_conf['patch_shape'] ## input patch size (32, 32, 8)
    sparse_scale = dataset_info['sparse_scale'] ## [1, 1, 4]
    if output_data is not None and trainTestFlag == 'train':

        # extraction_step is the same in training and test (4, 4, 4)
        extraction_step = train_test_conf['extraction_step']  ## shifting step (16, 16, 4)
        output_extraction_step = (modalities,) + tuple(np.array(extraction_step)*sparse_scale) ## (1, 16, 16, 16)
        output_shape = train_test_conf['output_shape'] ## (32, 32, 32)
        ## todo: judge if label_selector is necessary
        output_nominal_shape = tuple(np.array(patch_shape)*sparse_scale)
        output_selector = determine_output_selector(dimension, output_nominal_shape, output_shape)
        output_patch_shape = (modalities,) + output_shape ## (1, 32, 32, 32)
        output_patch = np.zeros((0,) + output_patch_shape, dtype=np.float32) ## output patch size

        minimum_non_bg = bg_discard_percentage * np.prod(output_shape)  ## number of non-background voxels in input patch
    else:
        output_patch = None
        extraction_step = train_test_conf['extraction_step_test']  ## shifting step (32, 32, 8)

    input_patch_shape = (modalities,) + patch_shape ## (1, 32, 32, 8)
    data_extraction_step = (modalities,) + extraction_step ## (1, 16, 16, 4)

    input_patch = np.zeros((0, ) + input_patch_shape, dtype=np.float32)
    len_input_data = len(input_data) # the first size of dimensions
    for idx in range(len_input_data):
        ## padding the original input image to make sure all patches can be extracted.
        pad_size = ()
        for dim in range(dimension):
            pad_size += (patch_shape[dim] // 2,)
        input_length = len(input_patch)

        if output_data is not None and trainTestFlag == 'train':
            ## padding the original output image to make sure all patches can be extracted
            output_pad_size = ()
            for dim in range(dimension):
                output_pad_size += (output_shape[dim] // 2,)
            output_length = len(output_patch)
            ## output
            output_vol = pad_both_sides(dimension, output_data[0], output_pad_size)
            output_data = np.delete(output_data, 0, 0)
            output_tmp_train = extract_patches(dimension, output_vol, output_patch_shape, output_extraction_step)
            del output_vol

            output_tmp_train = output_tmp_train[output_selector]
            ## dimension: patches x modalities x spatial size, considering modalities in this case
            sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)
            ## find indices of patches which has minimum backgrounds (0: background).
            valid_idxs = np.where(np.sum(output_tmp_train[:, representative_modality] != 0, axis=sum_axis) >= minimum_non_bg)
            N = max( valid_idxs[0].shape )

            output_patch = np.vstack((output_patch, np.zeros((N,) + output_patch_shape, dtype=np.float32)))

            output_patch[input_length:] = output_tmp_train[valid_idxs] # input length = output length
            del output_tmp_train

            # input
            input_vol = pad_both_sides(dimension, input_data[0], pad_size)
            input_data = np.delete(input_data, 0, 0)
            input_tmp_train = extract_patches(dimension, input_vol, input_patch_shape, data_extraction_step)
            del input_vol

            input_patch = np.vstack((input_patch, np.zeros((N,) + input_patch_shape, dtype=np.float32)))

            input_patch[input_length:] = input_tmp_train[valid_idxs]
            del input_tmp_train
        elif output_data is None and trainTestFlag == 'test':
            # no padding, no output selector, no overlapping
            ## todo: judge if the patches are overlapped.
            # input
            # input_vol = input_data[idx] # non-padding version
            input_vol = pad_both_sides(dimension, input_data[0], pad_size) # padding version
            input_data = np.delete(input_data, 0, 0)
            input_tmp_train = extract_patches(dimension, input_vol, input_patch_shape, data_extraction_step)
            del input_vol

            input_patch = np.vstack((input_patch, input_tmp_train))
            del input_tmp_train
        # else:
        #     ## todo: push error information

    ## shuffle patches and pick a subset
    
    # 20210126
    if output_data is not None and trainTestFlag == 'train':
        psr = train_test_conf['patch_sampling_rate']
        num_sampling_patch = int(np.ceil(input_patch.shape[0]*psr))
        smpl_patch_indices = np.arange(input_patch.shape[0])
        shuffle(smpl_patch_indices)
        smpl_patch_indices = smpl_patch_indices[:num_sampling_patch]

        input_patch = input_patch[smpl_patch_indices]
        output_patch = output_patch[smpl_patch_indices]
    

    return input_patch, output_patch
'''

'''
    Description: extract the central shape of output from patch
    patch_shape, output_shape should be odd in this problem
'''
def determine_output_selector(dimension, patch_shape, output_shape) :
    ndim = len(patch_shape)
    patch_shape_equal_output_shape = patch_shape == output_shape

    slice_none = slice(None)
    if not patch_shape_equal_output_shape :
        ## bug ?
        # return [slice_none] + [slice(output_shape[i], patch_shape[i] - output_shape[i]) for i in range(ndim)]
        return [slice_none] + [slice((patch_shape[i] - output_shape[i])//2, (patch_shape[i] + output_shape[i])//2) for i in range(ndim)]
    else :
        return [slice_none for i in range(ndim)]

def extract_patches(dimension, volume, patch_shape, extraction_step):
    actual_patch_shape = patch_shape
    actual_extraction_step = extraction_step
    # todo: need to check!!!
    if dimension == 2:
        if len(actual_patch_shape) == 3:
            actual_patch_shape = actual_patch_shape[:1] + (1,) + actual_patch_shape[1:]
            actual_extraction_step = actual_extraction_step[:1] + (1,) + actual_extraction_step[1:]
        else:
            actual_patch_shape = (1,) + actual_patch_shape
            actual_extraction_step = (1,) + actual_extraction_step

    patches = sk_extract_patches(
        volume,
        patch_shape=actual_patch_shape,
        extraction_step=actual_extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches,) + patch_shape)

def pad_both_sides(dimension, vol, pad) :
    pad_func = lambda vol, pad : np.pad(vol, pad, 'constant', constant_values=0)

    if dimension == 2 :
        pad = (0, ) + pad

    padding = ((pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2]))

    if len(vol.shape) == 3 :
        return pad_func(vol, padding)
    else :
        return pad_func(vol, ((0, 0),) + padding)


def visualise_patches(slices,
                      us,
                      save_name=None,
                      figsize=(6,6),
                      _vmin=0.0, _vmax=0.0015):
    """ Visualise 2d patches of uncertainty, etc (Tanno et al 2017)
    Args:
        x_slice (2d np.array):input
        y_slice (2d np.array):output
        us (int): upsampling rate
        figsize (tuple):figure size
    """

    fig, axes = plt.subplots(1, len(slices)+2, figsize=figsize)
    x_slice = slices[0]
    y_slice = slices[1]
    y_pred = slices[2]


    # input low-res patch x:
    axes[0].imshow(x_slice.T, cmap="gray", origin="lower", vmin=_vmin, vmax=_vmax)
    axes[0].set_title('input')
    inpN  = x_slice.shape[0]//2
    patch_radius = (y_slice.shape[0]//us)//2
    off =  inpN - patch_radius
    axes[0].add_patch(patches.Rectangle((off, off),
                                        2*patch_radius+1, 2*patch_radius+1,
                                        fill=False, edgecolor='red'))
    # input zoomed in:
    x_slice_zoom=x_slice[inpN-patch_radius:inpN+patch_radius+1, inpN-patch_radius:inpN+patch_radius+1]
    axes[1].imshow(x_slice_zoom.T, cmap="gray", origin="lower", vmin=_vmin, vmax=_vmax)
    axes[1].set_title('input (zoomed)')

    # ground truth output patch y:
    axes[2].imshow(y_slice.T, cmap="gray", origin="lower", vmin=_vmin, vmax=_vmax)
    axes[2].set_title('GT')

    # predicted output patch y:
    axes[3].imshow(y_pred.T, cmap="gray", origin="lower", vmin=_vmin, vmax=_vmax)
    axes[3].set_title('Prediction')

    # RMSE:
    rmse = np.sqrt((y_slice-y_pred)**2)
    axes[4].imshow(rmse.T, cmap="hot", origin="lower")
    axes[4].set_title('RMSE')

    # Uncertainty:
    if len(slices)>3:
        y_std = slices[3]
        axes[5].imshow(y_std.T, cmap="hot", origin="lower")
        axes[5].set_title('Uncertainty')

    #Save
    if not(save_name==None):
        fig.savefig(save_name, bbox_inches='tight')
        print("Saving "+ save_name)
