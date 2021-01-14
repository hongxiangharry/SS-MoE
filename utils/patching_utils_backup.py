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
from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches

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
    patch_shape = train_test_conf['patch_shape'] ## input patch size
    if output_data is not None and trainTestFlag == 'train':
        extraction_step = train_test_conf['extraction_step']  ## shifting step
        output_shape = train_test_conf['output_shape']
        ## todo: judge if label_selector is necessary
        output_selector = determine_output_selector(dimension, patch_shape, output_shape)
        output_patch_shape = (modalities,) + output_shape
        output_patch = np.zeros((0,) + output_patch_shape) ## output patch size
    else:
        output_patch = None
        extraction_step = train_test_conf['extraction_step_test']  ## shifting step

    minimum_non_bg = bg_discard_percentage * np.prod(patch_shape)

    input_patch_shape = (modalities,) + patch_shape
    data_extraction_step = (modalities,) + extraction_step

    input_patch = np.zeros((0, ) + input_patch_shape)
    len_input_data = len(input_data) # the first size of dimensions
    for idx in range(len_input_data):
        ## padding the original image to make sure all patches can be extracted.
        pad_size = ()
        for dim in range(dimension):
            pad_size += (patch_shape[dim] // 2,)
        input_length = len(input_patch)

        if output_data is not None and trainTestFlag == 'train':
            ## output
            output_vol = pad_both_sides(dimension, output_data[0], pad_size)
            output_data = np.delete(output_data, 0, 0)
            output_tmp_train = extract_patches(dimension, output_vol, input_patch_shape, data_extraction_step)
            del output_vol

            output_tmp_train = output_tmp_train[output_selector]
            ## dimension: patches x modalities x spatial size, considering modalities in this case
            sum_axis = (1, 2, 3) if dimension == 3 else (1, 2)
            ## find indices of patches which has minimum backgrounds (0: background).
            valid_idxs = np.where(np.sum(output_tmp_train[:, representative_modality] != 0, axis=sum_axis) >= minimum_non_bg)
            N = max( valid_idxs[0].shape )

            output_patch = np.vstack((output_patch, np.zeros((N,) + output_patch_shape)))

            output_patch[input_length:] = output_tmp_train[valid_idxs] # input length = output length
            del output_tmp_train

            # input
            input_vol = pad_both_sides(dimension, input_data[0], pad_size)
            input_data = np.delete(input_data, 0, 0)
            input_tmp_train = extract_patches(dimension, input_vol, input_patch_shape, data_extraction_step)
            del input_vol

            input_patch = np.vstack((input_patch, np.zeros((N,) + input_patch_shape)))

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
    return input_patch, output_patch


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