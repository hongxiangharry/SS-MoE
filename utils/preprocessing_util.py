'''

    Project: Self-supervised Mixture of Experts
    Publication: "Generalised Super Resolution for Quantitative MRI Using Self-supervised Mixture of Experts" published in MICCAI 2021.
    Authors: Hongxiang Lin, Yukun Zhou, Paddy J. Slator, Daniel C. Alexander
    Affiliation: Centre for Medical Image Computing, Department of Computer Science, University College London
    Email to the corresponding author: [Hongxiang Lin] harry.lin@ucl.ac.uk
    Date: 26/09/21
    Version: v1.0.1
    License: MIT

'''

import numpy as np
import nibabel as nib
import os
from nibabel.processing import resample_from_to, smooth_image

def preproc_dataset(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    if dataset == 'HCP-Wu-Minn-Contrast' :
        print('Downsample HCP data with 1D Gaussian filter.')
        preproc_in_idx = dataset_info['postfix_category']['preproc_in'] #
        in_postfix = dataset_info['postfix'][preproc_in_idx] # raw input data name
        preproc_out_idx = dataset_info['postfix_category']['preproc_out'] #
        out_postfix = dataset_info['postfix'][preproc_out_idx] # processed output data name
        return downsample_HCPWuMinnContrast_dataset(dataset_path, dataset_info, in_postfix, out_postfix)

def interp_input(gen_conf, test_conf, interp_order=3) :
    dataset = test_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    if dataset == 'Nigeria19-Multimodal':
        in_postfix = dataset_info['in_postfix'] # raw input data name
        out_postfix = dataset_info['interp_postfix'] # processed output data name
        return upsample_Nigeria19Multimodal_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = False, interp_order=interp_order)
    elif dataset == 'MBB':
        in_postfix = dataset_info['in_postfix'] # raw input data name
        out_postfix = dataset_info['interp_postfix'] # processed output data name
        return upsample_MBB_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = False, interp_order=interp_order)

def upsample_MBB_dataset(dataset_path,
                         dataset_info,
                         in_postfix,
                         out_postfix,
                         is_training = True,
                         interp_order = 3):
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    # in_postfix = dataset_info['postfix'][2] # raw input data name
    # out_postfix = dataset_info['postfix'][0] # processed output data name

    if is_training == True:
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    else:
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    voxel_scale = dataset_info['upsample_scale'] # upsample scale

    # in_data = np.zeros((num_volumes, modalities) + dimensions)
    # out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       in_postfix)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        out_postfix)
            # revise on 14/03/19
            # if (os.path.exists(out_filename) == False):
            data = nib.load(in_filename) # load raw data
            i_affine = np.dot(data.affine, np.diag(tuple(1.0/np.array(voxel_scale)) + (1.0, )))  # affine rescaling
            # i_shape = np.array(data.shape) * voxel_scale  # upsampled shape of output
            data = resample_from_to(data, (dimensions, i_affine), order=interp_order) # resize
            nib.save(data, out_filename)

    return True

def upsample_Nigeria19Multimodal_dataset(dataset_path,
                                         dataset_info,
                                         in_postfix,
                                         out_postfix,
                                         is_training = True,
                                         interp_order = 3):
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    prefix = dataset_info['prefix']
    # in_postfix = dataset_info['postfix'][2] # raw input data name
    # out_postfix = dataset_info['postfix'][0] # processed output data name
    ext = dataset_info['format']

    if is_training == True:
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    else:
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    voxel_scale = dataset_info['upsample_scale'] # upsample scale

    # in_data = np.zeros((num_volumes, modalities) + dimensions)
    # out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       prefix,
                                                       subject_lib[img_idx][:3],
                                                       modality_categories[mod_idx],
                                                       in_postfix,
                                                       ext)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        prefix,
                                                        subject_lib[img_idx][:3],
                                                        modality_categories[mod_idx],
                                                        out_postfix,
                                                        ext)

            # revise on 14/03/19
            # if (os.path.exists(out_filename) == False):
            data = nib.load(in_filename) # load raw data
            i_affine = np.dot(data.affine, np.diag(tuple(1.0/np.array(voxel_scale)) + (1.0, )))  # affine rescaling
            # i_shape = np.array(data.shape) * voxel_scale  # upsampled shape of output
            data = resample_from_to(data, (dimensions, i_affine), order=interp_order) # resize
            nib.save(data, out_filename)

    return True

def preproc_input(gen_conf, train_conf, is_training = True, interp_order=3) :
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    if dataset == 'Nigeria19-Multimodal':
        in_postfix = dataset_info['postfix'][2] # raw input data name
        out_postfix = dataset_info['postfix'][0] # processed output data name
        return upsample_HCPWuMinnContrast_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = is_training, interp_order=interp_order)

    if dataset == 'HCP-Wu-Minn-Contrast' :
        in_postfix = dataset_info['postfix'][2] # raw input data name
        out_postfix = dataset_info['postfix'][0] # processed output data name
        return upsample_HCPWuMinnContrast_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = is_training, interp_order=interp_order)

    if dataset == 'IBADAN-k8' :
        in_postfix = dataset_info['postfix'][2] # raw input data name
        out_postfix = dataset_info['postfix'][0] # processed output data name
        return upsample_HCPWuMinnContrast_dataset(dataset_path, dataset_info, in_postfix, out_postfix, is_training = is_training, interp_order=interp_order)

def downsample_HCPWuMinnContrast_dataset(dataset_path,
                                         dataset_info,
                                         in_postfix,
                                         out_postfix,
                                         voxel_scale = None):
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    # in_postfix = dataset_info['postfix'][2] # raw input data name
    # out_postfix = dataset_info['postfix'][0] # processed output data name

    subject_lib = dataset_info['training_subjects'] + dataset_info['test_subjects']
    num_volumes = dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]
    
    if voxel_scale is None:
        downsample_scale = dataset_info['downsample_scale']
        voxel_scale = [1, 1, downsample_scale] # downsample on an axial direction

    # in_data = np.zeros((num_volumes, modalities) + dimensions)
    # out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        out_postfix)
            print('Processing \''+in_filename+'\'')
            data = nib.load(in_filename) # load raw data
            fwhm = np.array(data.header.get_zooms()) * [0, 0, downsample_scale] # FWHM of Gaussian filter
            i_affine = np.dot(data.affine, np.diag(voxel_scale + [1]))  # affine rescaling
            i_shape = np.array(data.shape) // voxel_scale  # downsampled shape of output
            data = smooth_image(data, fwhm) # smoothed by FWHM
            data = resample_from_to(data, (i_shape, i_affine)) # resize
            nib.save(data, out_filename)
            print('Save to \''+out_filename+'\'')
            
    return True

def upsample_HCPWuMinnContrast_dataset(dataset_path,
                                       dataset_info,
                                       in_postfix,
                                       out_postfix,
                                       is_training = True,
                                       interp_order = 3):
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    # in_postfix = dataset_info['postfix'][2] # raw input data name
    # out_postfix = dataset_info['postfix'][0] # processed output data name

    if is_training == True:
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    else:
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    voxel_scale = dataset_info['upsample_scale'] # upsample scale

    # in_data = np.zeros((num_volumes, modalities) + dimensions)
    # out_data = np.zeros((num_volumes, modalities) + dimensions)

    for img_idx in range(num_volumes):
        for mod_idx in range(modalities):
            in_filename = os.path.join(dataset_path,
                                       path,
                                       pattern).format(subject_lib[img_idx],
                                                       modality_categories[mod_idx],
                                                       in_postfix)
            out_filename = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        modality_categories[mod_idx],
                                                        out_postfix)

            # revise on 14/03/19
            # if (os.path.exists(out_filename) == False):
            data = nib.load(in_filename) # load raw data
            i_affine = np.dot(data.affine, np.diag(tuple(1.0/np.array(voxel_scale)) + (1.0, )))  # affine rescaling
            # i_shape = np.array(data.shape) * voxel_scale  # upsampled shape of output
            data = resample_from_to(data, (dimensions, i_affine), order=interp_order) # resize
            nib.save(data, out_filename)

    return True
