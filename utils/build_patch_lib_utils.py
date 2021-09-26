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

import nibabel as nib
import numpy as np
import os
from architectures.arch_creator import generate_model
from utils.patching_utils import overlap_patching
from numpy.random import shuffle
import zipfile
import math
import shutil

def build_patch_lib(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    if dataset == 'MUDI' :
        return build_MUDI_patchset(gen_conf, train_conf)

def outlier_detection_indices_loader(path, n_subs, n_modalities):
    od_indices = np.loadtxt(path)
    od_indices = od_indices[:n_subs*n_modalities]
    od_indices = np.reshape(od_indices, (n_subs, n_modalities))
    return od_indices

def build_MUDI_patchset(gen_conf,
                        train_conf):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    patch_shape = train_conf['patch_shape']
    output_patch_shape = train_conf['output_shape']  # output patch shape
    extraction_step = train_conf['extraction_step']
    mnppf = train_conf['max_num_patches_per_file']  # per file

    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    real_modalities = dataset_info['real_modalities']
    modalities = dataset_info['modalities']
    path = dataset_info['path'][0] # process
    out_path = dataset_info['path'][3] # patch restored in the project storage
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']

    in_postfix = dataset_info['in_postfix']
    out_postfix = dataset_info['out_postfix']
    max_num_patches_per_vol = train_conf['max_num_patches'] # per vol

    subject_lib = dataset_info['training_subjects']
    num_volumes = dataset_info['num_volumes'][0]

    rebuild = train_conf['rebuild']

    # elif trainTestFlag == 'test':
    #     subject_lib = dataset_info['test_subjects']
    #     num_volumes = dataset_info['num_volumes'][1]
    # else:
    #     raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    out_patch_dir = os.path.join(dataset_path, out_path)
    if not os.path.isdir(out_patch_dir):
        os.makedirs(out_patch_dir)

    patch_filename_pattern = dataset_info['patch_filename_pattern']

    od_files = dataset_info['outlier_detection_files']
    od = train_conf['outlier_detection']
    if od is not None:
        if od == 'zscore' :
            od_path = os.path.join(dataset_path, path, od_files['z-score'])
            od_indices = outlier_detection_indices_loader(od_path, num_volumes, real_modalities)
        elif od == 'iqr' :
            od_path = os.path.join(dataset_path, path, od_files['iqr'])
            od_indices = outlier_detection_indices_loader(od_path, num_volumes, real_modalities)
        elif od == 'r-iqr' :
            od_path = os.path.join(dataset_path, path, od_files['iqr'])
            od_indices = 1-outlier_detection_indices_loader(od_path, num_volumes, real_modalities)
        else:
            raise ValueError("")

        out_patches_filename = patch_filename_pattern.format(in_postfix, patch_shape, extraction_step, num_volumes, od)
    else:
        od_indices = np.ones((num_volumes, real_modalities))
        out_patches_filename = patch_filename_pattern.format(in_postfix, patch_shape, extraction_step, num_volumes, 'None')

    os.chdir(out_patch_dir)
    if not os.path.isfile(out_patches_filename) or rebuild == True:
        count_patches = 0
        with zipfile.ZipFile(out_patches_filename, 'w') as z:
            for img_idx in range(num_volumes):
                # unzip source files
                in_zip_path = os.path.join(dataset_path,
                                           path,
                                           pattern).format(subject_lib[img_idx],
                                                           in_postfix, '', 'zip')
                in_unzip_dir = os.path.splitext(in_zip_path)[0]
                unzip(in_zip_path, in_unzip_dir)

                out_zip_path = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject_lib[img_idx],
                                                            out_postfix, '', 'zip')
                out_unzip_dir = os.path.splitext(out_zip_path)[0]
                unzip(out_zip_path, out_unzip_dir)

                # load source files
                for mod_idx in range(real_modalities):
                    # add outlier detection module, update [01/09/20]
                    if od_indices[img_idx, mod_idx] == 1:
                        in_filepath = os.path.join(dataset_path, path, subject_lib[img_idx], pattern).format(
                            in_postfix, in_postfix, str(mod_idx).zfill(4), 'nii.gz')
                        in_data = read_volume(in_filepath).astype(np.float32)
                        in_data = np.expand_dims(np.expand_dims(in_data, axis=0 ), axis=0)

                        out_filepath = os.path.join(dataset_path, path, subject_lib[img_idx], pattern).format(
                            out_postfix, out_postfix, str(mod_idx).zfill(4), 'nii.gz')
                        out_data = read_volume(out_filepath).astype(np.float32)
                        out_data = np.expand_dims(np.expand_dims(out_data, axis=0 ), axis=0)

                        # build patch lib
                        in_patches, out_patches = overlap_patching(gen_conf, train_conf, in_data, out_data)
                        # randomly some patches at most, update[31/08/20]
                        if max_num_patches_per_vol < len(in_patches):
                            random_order = np.random.choice(len(in_patches), max_num_patches_per_vol, replace=False)
                            in_patches = in_patches[random_order]
                            out_patches = out_patches[random_order]

                        count_patches += len(in_patches)
                        print("Have extracted {} patches ...".format(count_patches))
                        patch_filename_template = generate_patch_filename( mod_idx, subject_lib[img_idx], patch_shape, extraction_step, "{:04d}")

                        for sub_idx in range(math.ceil(len(in_patches) / mnppf)):
                            in_patch = in_patches[sub_idx * mnppf:np.minimum((sub_idx + 1) * mnppf, len(in_patches))]
                            out_patch = out_patches[sub_idx * mnppf:np.minimum((sub_idx + 1) * mnppf, len(out_patches))]
                            patch_filename = patch_filename_template.format(sub_idx)
                            save_patch_data(in_patch, out_patch, patch_filename)
                            print("Zipping " + patch_filename + " ...")
                            z.write(patch_filename)
                            os.remove(patch_filename)

                    else:
                        patch_filename = generate_patch_filename(mod_idx, subject_lib[img_idx], patch_shape, extraction_step, "{:04d}")
                        print("Skipping " + patch_filename + " ...")

                # clear source files
                shutil.rmtree(in_unzip_dir)
                shutil.rmtree(out_unzip_dir)
    else:
        print("The file '{}' is existed... ".format(out_patches_filename))
        count_patches = None

    return count_patches

def __save_volume(volume, nifty_affine, filename, format) :
    img = None
    if format == 'nii' or format == 'nii.gz' :
        img = nib.Nifti1Image(volume.astype('float32'), nifty_affine) # uint8
    if format == 'analyze' :
        img = nib.analyze.AnalyzeImage(volume.astype('float32'), nifty_affine) # uint8
    nib.save(img, filename)

def read_volume(filename) :
    return read_volume_data(filename).get_data()

def read_volume_data(filename) :
    return nib.load(filename)

def generate_output_filename(
    path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension) :
#     file_pattern = '{}/{}/{:02}-{}-{}-{}-{}.{}'
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    print(file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension))
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)

'''
    Read patch data
'''
def read_patch_data(filepath) :
    files = np.load(filepath).files
    return files['x_patches'], files['y_patches']

'''
    Save patch data with the input-image file name but with the '.npz' postfix 
'''
def save_patch_data(x_patches, y_patches, filepath) :
    np.savez(filepath, x_patches=x_patches, y_patches=y_patches)
    return True

def generate_patch_filename( modality, sample_num, patch_shape, extraction_step, sub, extension = 'npz') :
    file_pattern = '{}-{}-{}-{}-{}.{}'
    print(file_pattern.format( modality, sample_num, patch_shape, extraction_step, sub, extension))
    return file_pattern.format( modality, sample_num, patch_shape, extraction_step, sub, extension)

def unzip(src_filename, dest_dir):
    with zipfile.ZipFile(src_filename) as z:
        z.extractall(path=dest_dir)