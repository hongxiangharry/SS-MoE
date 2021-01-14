# process iso downsampled data

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
import ants
import zipfile

from nilearn import plotting
from nipype import Node, Workflow
from nipype.interfaces.fsl import Split


def compute_statistics(input_data):
    '''
    input_data: 4D (modality, x, y, z)
    '''
    mean = np.mean(input_data)
    std = np.std(input_data)
    return mean, std


def normalise_set(input_data, mean, std):
    '''
    input_data: 3D (x, y, z)
    '''
    input_data -= mean
    input_data /= std
    return input_data


if __name__ == "__main__":
    # conf
    iso_size = (41, 46, 28)
    aniso_size = (82, 92, 28)
    original_size = (82, 92, 56)

    num_modalities = 1344

    dataset_dir = '/cluster/project0/IQT_Nigeria/others/SuperMudi/origin'
    proc_dataset_dir = '/cluster/project0/IQT_Nigeria/others/SuperMudi/process'
    raw_data_filename = 'aniso.nii.gz'
    gt_data_filename = 'org.nii.gz'
    brain_mask_filename = 'brain_mask.nii.gz'
    # MB_Re_t_moco_registered_applytopup_anisotropic_voxcor.nii.gz # aniso-lr
    # MB_Re_t_moco_registered_applytopup_resized.nii.gz # original

    # search sub-folder having GT
    subs = sorted([os.path.basename(os.path.dirname(path)) for path in glob.glob(dataset_dir + '/cdmri*/' + gt_data_filename)])

    # prefix of data type (iso-lr/aniso-lr/origin)
    proc_data_base_name = 'interp-aniso.nii.gz'

    # interp order
    interp_order = 3

    # mkdir
    for sub in subs:
        proc_dir = os.path.join(proc_dataset_dir, sub)
        if os.path.isdir(proc_dir) is False:
            os.makedirs(proc_dir)

    # # resample from aniso to origin
    # print("Script for interpolating Aniso data ... ")
    # print("Re-sampling data ... ")
    # for sub in subs:
    #     print("Resampling {} ...".format(sub))
    #     raw_data_path = os.path.join(dataset_dir, sub, raw_data_filename)
    #     gt_data_path = os.path.join(dataset_dir, sub, gt_data_filename)
    #
    #     raw_handle = nib.load(filename=raw_data_path)
    #     gt_handle = nib.load(filename=gt_data_path)
    #
    #     interp_handle = resample_from_to(raw_handle, gt_handle, order=interp_order)
    #     proc_data_path = os.path.join(proc_dataset_dir, sub, proc_data_base_name)
    #     nib.save(interp_handle, proc_data_path)

    # register the brain mask to the FOV of GT
    for sub in subs:
        gt_data_path = os.path.join(dataset_dir, sub, gt_data_filename)
        proc_data_path = os.path.join(proc_dataset_dir, sub, proc_data_base_name)
        brain_mask_path = os.path.join(dataset_dir, sub, brain_mask_filename)
        proc_brain_mask_path = os.path.join(proc_dataset_dir, sub, brain_mask_filename)

        gt_shape = nib.load(filename=gt_data_path).shape

        mask_handle = nib.load(filename=brain_mask_path)
        mask_volume = mask_handle.get_fdata()[:gt_shape[0], :gt_shape[1], :gt_shape[2]]
        mask_handle = nib.Nifti1Image(mask_volume, mask_handle.affine)

        nib.save(mask_handle, proc_brain_mask_path)

    # ## compute mse
    # mse = 0
    # for sub in subs:
    #     gt_data_path = os.path.join(dataset_dir, sub, gt_data_filename)
    #     proc_data_path = os.path.join(proc_dataset_dir, sub, proc_data_base_name)
    #     proc_brain_mask_path = os.path.join(proc_dataset_dir, sub, brain_mask_filename)
    #
    #     gt_volume = nib.load(filename=gt_data_path).get_fdata() # (x,y,z,t)
    #     interp_volume = nib.load(filename=proc_data_path).get_fdata() # (x,y,z,t)
    #     mask_volume = nib.load(filename=proc_brain_mask_path).get_fdata() # (x,y,z)
    #     mask_volume = np.expand_dims(mask_volume, axis=3)
    #
    #     mse_sub = np.mean((gt_volume-interp_volume)**2 * mask_volume)
    #     print("The MSE of the subject {} is {}".format(sub, mse_sub))
    #     mse += mse_sub
    #
    # mse = mse/len(subs)
    #
    # print("The sum of MSE over all subjects is {}".format(mse))
