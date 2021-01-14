# process aniso downsampled data

import os
import glob
import numpy as np
import nibabel as nib
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
    raw_data_filename = 'aniso.nii.gz' # aniso-lr
    brain_mask_filename = 'brain_mask.nii.gz' # brain mask
    # 'MB_Re_t_moco_registered_applytopup_isotropic_voxcor.nii.gz' 
    # MB_Re_t_moco_registered_applytopup_anisotropic_voxcor.nii.gz # aniso-lr
    # MB_Re_t_moco_registered_applytopup_resized.nii.gz # original

    # search sub-folder
    subs = [os.path.basename(os.path.dirname(path)) for path in glob.glob(dataset_dir+'/cdmri*/'+raw_data_filename)]

    # prefix of data type (iso-lr/aniso-lr/origin)
    proc_data_base_name = 'aniso'
    image_size = aniso_size

    # mkdir
    for sub in subs:
        proc_dir = os.path.join(proc_dataset_dir, sub)
        if os.path.isdir(proc_dir) is False:
            os.makedirs(proc_dir)
    
    # # split 4d nifty to 3d
    # print("Split 4D nifty...")
    # for sub in subs:
    #     raw_data_path = os.path.join(dataset_dir, sub, raw_data_filename)
    #     sub_path = os.path.join(proc_dataset_dir, sub)
    #     os.chdir(sub_path) # go to the 'sub_path' dir
    #     Split(in_file=raw_data_path, dimension='t', out_base_name=proc_data_base_name).run()
    
    # # pad images
    # print("Pad images...")
    # for sub in subs:
    #     proc_paths = glob.glob(os.path.join(proc_dataset_dir, sub, proc_data_base_name)+'*.nii.gz')
    #     for proc_path in proc_paths:
    #         print(proc_path)
    #         mri_handle = nib.load(filename=proc_path)
    #
    #         mri_volume = np.zeros(image_size)
    #         mri_volume[:mri_handle.shape[0], :mri_handle.shape[1], :mri_handle.shape[2]] = mri_handle.get_fdata()
    #         mri_handle = nib.Nifti1Image(mri_volume, mri_handle.affine)
    #
    #         nib.save(mri_handle, proc_path)
            
    ## standardization
    print("Standardization...")
    mean_arr = []
    std_arr = []
    for modality in range(num_modalities):
        input_data = np.zeros((len(subs),) + image_size)
        for idx, sub in enumerate(subs):
            proc_data_path = os.path.join(proc_dataset_dir, sub, proc_data_base_name+str(modality).zfill(4)+'.nii.gz')
            input_data[idx] = nib.load(proc_data_path).get_fdata()
        mean, std = compute_statistics(input_data)
        mean_arr.append(mean)
        std_arr.append(std)

        for sub in subs:
            proc_data_path = os.path.join(proc_dataset_dir, sub, proc_data_base_name+str(modality).zfill(4)+'.nii.gz')
            mri_handle = nib.load(proc_data_path)
            mri_volume = mri_handle.get_fdata()
            mri_volume = normalise_set(mri_volume, mean, std)
            mri_handle = nib.Nifti1Image(mri_volume, mri_handle.affine)
            nib.save(mri_handle, sub)

    ## save all mean and std arrays
    np.save(os.path.join(proc_dataset_dir,proc_data_base_name+'_mean'), mean_arr)
    np.save(os.path.join(proc_dataset_dir,proc_data_base_name+'_std'), std_arr)

    print("Zipping images...")
    for sub in subs:
        sub_path = os.path.join(proc_dataset_dir, sub)
        os.chdir(sub_path)
        with zipfile.ZipFile(os.path.join(proc_data_base_name+'.zip'), 'w') as z:
            modality_paths = glob.glob(proc_data_base_name+'*.nii.gz')
            for modal_path in modality_paths:
                print("Zipping " + modal_path)
                z.write(modal_path)
                os.remove(modal_path)

