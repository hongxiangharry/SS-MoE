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

def compute_statistics(input_data) :
    '''
    input_data: 4D (modality, x, y, z)
    '''
    mean = np.mean(input_data)
    std = np.std(input_data)
    return mean, std

def normalise_set(input_data, mean, std) :
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
    interp_data_filename = 'interp-iso.nii.gz'
    gt_data_filename = 'org.nii.gz'
    brain_mask_filename = 'brain_mask.nii.gz'

    mse_save_filename = 'iso_interp_mse_array.txt'

    zscore_filename = 'iso-interp_zscore.txt'
    zscore_th = 2.5
    iqr_filename = 'iso-interp_iqr.txt'

    # MB_Re_t_moco_registered_applytopup_anisotropic_voxcor.nii.gz # aniso-lr
    # MB_Re_t_moco_registered_applytopup_resized.nii.gz # original

    # search sub-folder having GT
    subs = [os.path.basename(os.path.dirname(path)) for path in sorted(glob.glob(dataset_dir+'/cdmri*/'+gt_data_filename))]

    # interp order
    interp_order = 3

    ## compute MSE per volume
    mse_array = []
    for sub in subs:
        gt_data_path = os.path.join(dataset_dir, sub, gt_data_filename)
        interp_data_path = os.path.join(proc_dataset_dir, sub, interp_data_filename)
        brain_mask_path = os.path.join(proc_dataset_dir, sub, brain_mask_filename)

        gt_volume = nib.load(filename=gt_data_path).get_fdata() # (x,y,z,t)
        interp_volume = nib.load(filename=interp_data_path).get_fdata() # (x,y,z,t)
        mask_volume = nib.load(filename=brain_mask_path).get_fdata() # (x,y,z)

        for idx_vol in range(gt_volume.shape[3]):
            mse = np.mean((gt_volume[:, :, :, idx_vol]-interp_volume[:, :, :, idx_vol])**2 * mask_volume)
            print("The MSE of the subject {}, volume {}th is {}".format(sub, idx_vol, mse))
            mse_array.append(mse)
    mse_array = np.array(mse_array)
    mse_save_path = os.path.join(proc_dataset_dir, mse_save_filename)
    np.savetxt(mse_save_path, mse_array)

    # outlier detection
    # zscore:
    zscore = np.array( (mse_array-np.mean(mse_array))/np.std(mse_array) <= zscore_th ).astype(int)
    zscore_save_path = os.path.join(proc_dataset_dir, zscore_filename)
    np.savetxt(zscore_save_path, zscore)
    # iqr
    q1 = np.quantile(mse_array, 0.25)
    q3 = np.quantile(mse_array, 0.75)
    iqr = (mse_array >= q1) & (mse_array < q3)
    iqr_save_path = os.path.join(proc_dataset_dir, iqr_filename)
    np.savetxt(iqr_save_path, iqr)