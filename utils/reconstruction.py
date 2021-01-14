import itertools
from utils.patching_utils import pad_both_sides
from utils.patching_utils import determine_output_selector
import numpy as np

from .general_utils import pad_both_sides

def reconstruct_volume_imaging4(gen_conf, train_conf, patches) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = train_conf['dimension'] ## 3
    expected_shape = dataset_info['dimensions'] # output image size (260, 311, 260)
    extraction_step = train_conf['extraction_step_test'] # shifting step (16, 16, 2)
    output_shape = train_conf['output_shape_test'] # output patch size (16, 16, 16)
    patch_shape = train_conf['patch_shape'] # input patch size (32, 32, 4)
    sparse_scale = dataset_info['sparse_scale']  ## [1, 1, 8]

    if dimension == 2 :
        output_shape = (0, ) + output_shape if dimension == 2 else output_shape
        extraction_step = (1, ) + extraction_step if dimension == 2 else extraction_step

    # output selector : 21/3/19
    output_extraction_step = tuple(np.array(extraction_step) * sparse_scale)
    output_nominal_shape = tuple(np.array(patch_shape) * sparse_scale)
    output_selector = determine_output_selector(dimension, output_nominal_shape, output_shape)

    # padding
    output_pad_size = ()
    output_pad_expected_shape = ()
    for dim in range(dimension):
        output_pad_size += (output_shape[dim] // 2,) ## (32, 32, 32)
        output_pad_expected_shape += (expected_shape[dim]+output_shape[dim],) # padding image size (292, 343, 292)

    rec_volume = np.zeros(expected_shape)
    rec_volume = pad_both_sides(dimension, rec_volume, output_pad_size) # padding
    rec_patch_count = np.zeros(expected_shape)
    rec_patch_count = pad_both_sides(dimension, rec_patch_count, output_pad_size) # padding

    output_patch_volume = np.ones(output_shape) * 1.0

    coordinates = generate_indexes(
        dimension, output_shape, output_extraction_step, output_pad_expected_shape)

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))] # non-padding
        ## rec_volume[selection] += patches[count] # non-padding
        rec_volume[selection] += patches[output_selector][count]   ## non-padding
        rec_patch_count[selection] += output_patch_volume
    # overlapping reconstruction: average patch
    rec_volume = rec_volume/((rec_patch_count == 0)+rec_patch_count)

    # un-padding: 3D
    rec_volume = rec_volume[output_pad_size[0]:-output_pad_size[0],
                 output_pad_size[1]:-output_pad_size[1],
                 output_pad_size[2]:-output_pad_size[2]]
    return rec_volume

def reconstruct_volume_imaging3(gen_conf, train_conf, patches) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = train_conf['dimension'] ## 3
    expected_shape = dataset_info['dimensions'] # output image size (260, 311, 260)
    extraction_step = train_conf['extraction_step_test'] # shifting step (32, 32, 8)
    output_shape = train_conf['output_shape'] # output patch size (32, 32, 32)
    patch_shape = train_conf['patch_shape'] # input patch size (32, 32, 8)
    sparse_scale = dataset_info['sparse_scale']  ## [1, 1, 4]

    if dimension == 2 :
        output_shape = (0, ) + output_shape if dimension == 2 else output_shape
        extraction_step = (1, ) + extraction_step if dimension == 2 else extraction_step

    output_extraction_step = tuple(np.array(extraction_step) * sparse_scale)
    # padding
    output_pad_size = ()
    output_pad_expected_shape = ()
    for dim in range(dimension):
        output_pad_size += (output_shape[dim] // 2,) ## (32, 32, 32)
        output_pad_expected_shape += (expected_shape[dim]+output_shape[dim],) # padding image size (292, 343, 292)

    rec_volume = np.zeros(expected_shape)
    rec_volume = pad_both_sides(dimension, rec_volume, output_pad_size) # padding
    rec_patch_count = np.zeros(expected_shape)
    rec_patch_count = pad_both_sides(dimension, rec_patch_count, output_pad_size) # padding

    output_patch_volume = np.ones(output_shape) * 1.0

    coordinates = generate_indexes(
        dimension, output_shape, output_extraction_step, output_pad_expected_shape)

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))] # non-padding
        ## rec_volume[selection] += patches[count] # non-padding
        rec_volume[selection] += patches[count]   ## non-padding
        rec_patch_count[selection] += output_patch_volume
    # overlapping reconstruction: average patch
    rec_volume = rec_volume/((rec_patch_count == 0)+rec_patch_count)

    # un-padding: 3D
    rec_volume = rec_volume[output_pad_size[0]:-output_pad_size[0],
                 output_pad_size[1]:-output_pad_size[1],
                 output_pad_size[2]:-output_pad_size[2]]
    return rec_volume

# imaging
def reconstruct_volume_imaging(gen_conf, train_conf, patches) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = train_conf['dimension']
    expected_shape = dataset_info['dimensions'] # output image size
    extraction_step = train_conf['extraction_step_test'] # shifting step
    output_shape = train_conf['output_shape'] # output patch size
    patch_shape = train_conf['patch_shape'] # input patch size

    if dimension == 2 :
        output_shape = (0, ) + output_shape if dimension == 2 else output_shape
        extraction_step = (1, ) + extraction_step if dimension == 2 else extraction_step
    # padding
    pad_size = ()
    pad_expected_shape = ()
    for dim in range(dimension):
        pad_size += (patch_shape[dim] // 2,)
        pad_expected_shape += (expected_shape[dim]+patch_shape[dim],) # padding image size

    rec_volume = np.zeros(expected_shape)
    rec_volume = pad_both_sides(dimension, rec_volume, pad_size) # padding
    rec_patch_count = np.zeros(expected_shape)
    rec_patch_count = pad_both_sides(dimension, rec_patch_count, pad_size) # padding

    output_patch_volume = np.ones(output_shape) * 1.0

    coordinates = generate_indexes(
        dimension, output_shape, extraction_step, pad_expected_shape)

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))] # non-padding
        ## rec_volume[selection] += patches[count] # non-padding
        rec_volume[selection] += patches[count]  # non-padding
        rec_patch_count[selection] += output_patch_volume
    # overlapping reconstruction: average patch
    rec_volume = rec_volume/((rec_patch_count == 0)+rec_patch_count)

    # un-padding: 3D
    rec_volume = rec_volume[pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1], pad_size[2]:-pad_size[2]]
    return rec_volume

# segmentation
def reconstruct_volume(gen_conf, train_conf, patches) :
    dataset = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dimension = train_conf['dimension']
    expected_shape = dataset_info['dimensions']
    extraction_step = train_conf['extraction_step_test']
    num_classes = gen_conf['num_classes']
    output_shape = train_conf['output_shape']
    patch_shape = train_conf['patch_shape']

    if dimension == 2 :
        output_shape = (0, ) + output_shape if dimension == 2 else output_shape
        extraction_step = (1, ) + extraction_step if dimension == 2 else extraction_step

    rec_volume = perform_voting(
        dimension, patches, output_shape, expected_shape, extraction_step, num_classes)

    return rec_volume
# segmentation
def perform_voting(dimension, patches, output_shape, expected_shape, extraction_step, num_classes) :
    vote_img = np.zeros(expected_shape + (num_classes, ))

    coordinates = generate_indexes(
        dimension, output_shape, extraction_step, expected_shape)

    if dimension == 2 : 
        output_shape = (1, ) + output_shape[1:]

    for count, coord in enumerate(coordinates) :
        selection = [slice(coord[i] - output_shape[i], coord[i]) for i in range(len(coord))]
        selection += [slice(None)]
        vote_img[selection] += patches[count]

    #return np.argmax(vote_img[:, :, :, 1:], axis=3) + 1
    return np.argmax(vote_img, axis=3)

def generate_indexes(dimension, output_shape, extraction_step, expected_shape) :
    # expected_shape: total size
    ndims = len(output_shape)

    poss_shape = [output_shape[i] + extraction_step[i] * ((expected_shape[i] - output_shape[i]) // extraction_step[i]) for i in range(ndims)]

    if dimension == 2 :
        output_shape = (1, ) + output_shape[1:]

    idxs = [range(output_shape[i], poss_shape[i] + 1, extraction_step[i]) for i in range(ndims)]
    
    return itertools.product(*idxs)