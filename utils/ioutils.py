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
import shutil

def read_meanstd(gen_conf, test_conf, case_name = 1) :
    mean_filename = generate_output_filename(
            gen_conf['model_path'],
            test_conf['dataset'],
            case_name,
            test_conf['approach'],
            test_conf['dimension'],
            str(test_conf['patch_shape']),
            str(test_conf['extraction_step'])+'_mean',
            'npz')
#     with open(mean_filename, mode='r') as infile:
#         reader = csv.reader(infile)
#         mean = {rows[0]:np.array(rows[1]) for rows in reader}
    mean = {}
    mean_f = np.load(mean_filename)
    mean['input'] = mean_f['mean_input']
    mean['output'] = mean_f['mean_output']
        
    std_filename = generate_output_filename(
            gen_conf['model_path'],
            test_conf['dataset'],
            case_name,
            test_conf['approach'],
            test_conf['dimension'],
            str(test_conf['patch_shape']),
            str(test_conf['extraction_step'])+'_std',
            'npz')
#     with open(std_filename, mode='r') as infile:
#         reader = csv.reader(infile)
#         std = {rows[0]:np.array(rows[1]) for rows in reader}
    std = {}
    std_f = np.load(std_filename)
    std['input'] = std_f['std_input']
    std['output'] = std_f['std_output']
    return mean, std

def save_meanstd(gen_conf, train_conf, mean, std, case_name = 1):
    ## save mean and std
    mean_filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']) + '_mean',
        'npz')
    std_filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']) + '_std',
        'npz')
    ## check and make folders
    meanstd_foldername = os.path.dirname(mean_filename)
    if not os.path.isdir(meanstd_foldername):
        os.makedirs(meanstd_foldername)

    if (mean is None) or (std is None):
        mean = {'input': np.array([0.0]), 'output': np.array([0.0])}
        std = {'input': np.array([1.0]), 'output': np.array([1.0])}
    np.savez(mean_filename, mean_input=mean['input'], mean_output=mean['output'])
    np.savez(std_filename, std_input=std['input'], std_output=std['output'])
    return True

def read_meanstd_MUDI_output(gen_conf, test_conf):
    dataset = test_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    path = dataset_info['path'][0]
    pattern = dataset_info['general_pattern']

    mean_filename = os.path.join(dataset_path, path, 'org_mean.npy')

    mean = np.load(mean_filename)

    std_filename = os.path.join(dataset_path, path, 'org_std.npy')
    std = np.load(std_filename)
    return mean, std

def save_random_samples(gen_conf, train_conf, ran_samples, case_name = 1):
    filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']) + '_random_samples',
        'npz')
    ## check and make folders
    foldername = os.path.dirname(filename)
    if not os.path.isdir(foldername):
        os.makedirs(foldername)

    np.savez(filename, ran_samples=ran_samples)
    return True

def read_msecorr_array(gen_conf, train_conf) :
    filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        'mse_corr',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'npz')
    data = np.load(filename)
    return data['mse_array'], data['corr_array']

def save_msecorr_array(gen_conf, train_conf, mse_array, corr_array) :
    filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        'mse_corr',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'npz')
    ## check and make folders
    foldername = os.path.dirname(filename)
    if not os.path.isdir(foldername):
        os.makedirs(foldername)
    np.savez(filename, mse_array=mse_array, corr_array=corr_array)
    return True


def save_msecorr_array_S2(gen_conf, train_conf, mse_array, corr_array) :
    filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        'mse_corr_S2',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'npz')
    ## check and make folders
    foldername = os.path.dirname(filename)
    if not os.path.isdir(foldername):
        os.makedirs(foldername)
    np.savez(filename, mse_array=mse_array, corr_array=corr_array)
    return True



def read_model(gen_conf, train_conf, case_name) :
    model = generate_model(gen_conf, train_conf)

    model_filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'h5')

    print('!!!!!!!!!!!', model_filename)
    model.load_weights(model_filename)

    return model

def read_dataset(gen_conf, train_conf, trainTestFlag = 'train') :
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    if dataset == 'MUDI' :
        #return read_MUDI_dataset(dataset_path, dataset_info, trainTestFlag)
        return read_MUDI_dataset_2(dataset_path, dataset_info, trainTestFlag)


def read_MUDI_dataset_2(dataset_path,
                      dataset_info,
                      trainTestFlag='train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    real_modalities = dataset_info['real_modalities']
    modalities = dataset_info['modalities']
    path = dataset_info['path'][0] # process, 0
    unzip_path = dataset_info['path'][2] # patch, 2
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['in_postfix']
    out_postfix = dataset_info['out_postfix']

    filename_list = []

    if trainTestFlag == 'train':
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    elif trainTestFlag == 'test' or trainTestFlag == 'eval':
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else:
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes * real_modalities, 1) + input_dimension, dtype=np.float32)
    if trainTestFlag == 'train' or trainTestFlag == 'eval':
        out_data = np.zeros((num_volumes * real_modalities, 1) + dimensions, dtype=np.float32)
    else:
        out_data = None


    # print out to see which represent the patch name
    for img_idx in range(num_volumes):
        in_zip_name = os.path.join(dataset_path,
                                   path,
                                   pattern).format(subject_lib[img_idx],
                                                   in_postfix, '', 'zip')
        in_unzip_dir = os.path.splitext(os.path.join(dataset_path, unzip_path, pattern)
                                        .format(subject_lib[img_idx], in_postfix+'_unzip', '', 'zip'))[0]
        unzip(in_zip_name, in_unzip_dir)



        if trainTestFlag == 'train' or trainTestFlag == 'eval':
            out_zip_name = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        out_postfix, '', 'zip')
            out_unzip_dir = os.path.splitext(os.path.join(dataset_path, unzip_path, pattern)
                                        .format(subject_lib[img_idx], out_postfix+'_unzip', '', 'zip'))[0]
            unzip(out_zip_name, out_unzip_dir)

        for mod_idx in range(real_modalities):
            in_filename = os.path.join(dataset_path, unzip_path, subject_lib[img_idx], pattern).format(in_postfix+'_unzip', in_postfix, str(mod_idx).zfill(4), 'nii.gz')
            # maybe this in_filename & split is good for the name list
            
            #print('the in_filename is: ', in_filename)
            in_data[img_idx * real_modalities + mod_idx, 0] = read_volume(in_filename).astype(np.float32)

            filename_list.append(in_filename)

            if trainTestFlag == 'train' or trainTestFlag == 'eval':
                out_filename = os.path.join(dataset_path, unzip_path, subject_lib[img_idx], pattern).format(out_postfix+'_unzip', out_postfix, str(mod_idx).zfill(4), 'nii.gz')

                #print('the out_filename is: ', out_filename)

                out_data[img_idx * real_modalities + mod_idx, 0] = read_volume(out_filename).astype(np.float32)

        shutil.rmtree(in_unzip_dir)
        if trainTestFlag == 'train' or trainTestFlag == 'eval':
            shutil.rmtree(out_unzip_dir)

    print('111111111111', np.shape(filename_list))
    # return the name list out
    return in_data, out_data, filename_list




def read_MUDI_dataset(dataset_path,
                      dataset_info,
                      trainTestFlag='train'):
    dimensions = dataset_info['dimensions']
    sparse_scale = dataset_info['sparse_scale']
    input_dimension = tuple(np.array(dimensions) // sparse_scale)
    real_modalities = dataset_info['real_modalities']
    modalities = dataset_info['modalities']
    path = dataset_info['path'][0] # process, 0
    unzip_path = dataset_info['path'][2] # patch, 2
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['in_postfix']
    out_postfix = dataset_info['out_postfix']
    if trainTestFlag == 'train':
        subject_lib = dataset_info['training_subjects']
        num_volumes = dataset_info['num_volumes'][0]
    elif trainTestFlag == 'test' or trainTestFlag == 'eval':
        subject_lib = dataset_info['test_subjects']
        num_volumes = dataset_info['num_volumes'][1]
    else:
        raise ValueError("trainTestFlag should be declared as 'train'/'test'/'evaluation'")

    in_data = np.zeros((num_volumes * real_modalities, 1) + input_dimension, dtype=np.float32)
    if trainTestFlag == 'train' or trainTestFlag == 'eval':
        out_data = np.zeros((num_volumes * real_modalities, 1) + dimensions, dtype=np.float32)
    else:
        out_data = None


    # print out to see which represent the patch name
    for img_idx in range(num_volumes):
        in_zip_name = os.path.join(dataset_path,
                                   path,
                                   pattern).format(subject_lib[img_idx],
                                                   in_postfix, '', 'zip')
        in_unzip_dir = os.path.splitext(os.path.join(dataset_path, unzip_path, pattern)
                                        .format(subject_lib[img_idx], in_postfix+'_unzip', '', 'zip'))[0]
        unzip(in_zip_name, in_unzip_dir)



        if trainTestFlag == 'train' or trainTestFlag == 'eval':
            out_zip_name = os.path.join(dataset_path,
                                        path,
                                        pattern).format(subject_lib[img_idx],
                                                        out_postfix, '', 'zip')
            out_unzip_dir = os.path.splitext(os.path.join(dataset_path, unzip_path, pattern)
                                        .format(subject_lib[img_idx], out_postfix+'_unzip', '', 'zip'))[0]
            unzip(out_zip_name, out_unzip_dir)

        for mod_idx in range(real_modalities):
            in_filename = os.path.join(dataset_path, unzip_path, subject_lib[img_idx], pattern).format(in_postfix+'_unzip', in_postfix, str(mod_idx).zfill(4), 'nii.gz')
            # maybe this in_filename & split is good for the name list
            
            print('the in_filename is: ', in_filename)
            in_data[img_idx * real_modalities + mod_idx, 0] = read_volume(in_filename).astype(np.float32)
            if trainTestFlag == 'train' or trainTestFlag == 'eval':
                out_filename = os.path.join(dataset_path, unzip_path, subject_lib[img_idx], pattern).format(out_postfix+'_unzip', out_postfix, str(mod_idx).zfill(4), 'nii.gz')

                print('the out_filename is: ', out_filename)

                out_data[img_idx * real_modalities + mod_idx, 0] = read_volume(out_filename).astype(np.float32)

        shutil.rmtree(in_unzip_dir)
        if trainTestFlag == 'train' or trainTestFlag == 'eval':
            shutil.rmtree(out_unzip_dir)

    # return the name list out
    return in_data, out_data


def save_volume(gen_conf, test_conf, volume, case_idx, flag = 'test') :
    dataset = test_conf['dataset']
    if dataset == 'MUDI':
        return save_volume_MUDI(gen_conf, test_conf, volume, case_idx, flag=flag)
    else:
        return save_volume_else(gen_conf, test_conf, volume, case_idx)

def save_volume_MUDI(gen_conf, test_conf, volume, case_idx, flag='test') :
    '''
    save as 4D NIFTY
    :param gen_conf:
    :param test_conf:
    :param volume: 4D (x,y,z,t)
    :param case_idx: 0:subject, 1: case
    :return:
    '''
    # todo: check conf and modalities
    dataset = test_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path'][1] # origin
    pattern = dataset_info['general_pattern']
    modality_categories = dataset_info['modality_categories']
    in_postfix = dataset_info['in_postfix']
    if flag == 'test' or flag == 'eval':
        subject_lib = dataset_info['test_subjects']
    elif flag == 'train':
        subject_lib = dataset_info['training_subjects']
    sparse_scale = dataset_info['sparse_scale']

    data_filename = os.path.join(dataset_path, path, pattern)\
        .format(subject_lib[case_idx[0]], in_postfix, '', 'nii.gz')
    image_data = read_volume_data(data_filename)

    ## change affine in the nii header
    if sparse_scale is not None:
        assert len(sparse_scale) == 3, "The length of sparse_scale is not equal to 3."
        # print image_data.affine
        nifty_affine = np.dot(image_data.affine, np.diag(tuple(1.0/np.array(sparse_scale))+(1.0, )))
        # print nifty_affine

    ## save as 4D nifty
    out_filename = generate_output_filename(
        gen_conf['results_path'],
        test_conf['dataset'],
        subject_lib[case_idx[0]]+'_c'+str(case_idx[1]),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        dataset_info['format'])
    out_foldername = os.path.dirname(out_filename)
    if not os.path.isdir(out_foldername) :
        os.makedirs(out_foldername)
    print("Save file at {}".format(out_filename))
    __save_volume(volume, nifty_affine, out_filename, dataset_info['format'])

def save_volume_else(gen_conf, train_conf, volume, case_idx) :
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    extraction_step = train_conf['extraction_step_test']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    if dataset == 'iSeg2017' or dataset == 'IBSR18':
        volume_tmp = np.zeros(volume.shape + (1, ))
        volume_tmp[:, :, :, 0] = volume
        volume = volume_tmp

    data_filename = dataset_path + path + pattern.format(case_idx, inputs[-1])
    image_data = read_volume_data(data_filename)

    volume = np.multiply(volume, image_data.get_data() != 0)

    if dataset == 'iSeg2017' :
        volume[image_data.get_data() != 0] = volume[image_data.get_data() != 0] + 1

        label_mapper = {0 : 0, 1 : 10, 2 : 150, 3 : 250}
        for key in label_mapper.keys() :
            volume[volume == key] = label_mapper[key]

    out_filename = results_path + path + pattern.format(case_idx, approach + ' - ' + str(extraction_step))

    ## mkdir
    if not os.path.isdir(os.path.dirname(out_filename)):
        os.makedirs(os.path.dirname(out_filename))

    __save_volume(volume, image_data.affine, out_filename, dataset_info['format'])

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
#     file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
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

def generate_patch_filename( modality, sample_num, patch_shape, extraction_step, extension = 'npz') :
    file_pattern = '{}-{}-{}-{}.{}'
    print(file_pattern.format( modality, sample_num, patch_shape, extraction_step, extension))
    return file_pattern.format( modality, sample_num, patch_shape, extraction_step, extension)

def unzip(src_filename, dest_dir):
    with zipfile.ZipFile(src_filename) as z:
        z.extractall(path=dest_dir)