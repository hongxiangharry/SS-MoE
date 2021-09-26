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
import os
import tensorflow as tf
import pandas as pd
from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks
from utils.ioutils import read_dataset, save_volume, read_model, generate_output_filename, read_meanstd, read_meanstd_MUDI_output
from utils.reconstruction import reconstruct_volume_imaging, reconstruct_volume_imaging4, reconstruct_volume_imaging3
from utils.patching_utils import overlap_patching, overlap_patching_test
from utils.preprocessing_util import preproc_input

def evaluating_S2_test_label(gen_conf, train_conf, flag = 'train', case_name = 0) :
    '''
    :param gen_conf:
    :param train_conf:
    :param flag: train/eval
    :param case_name:
    :return:
    '''
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]

    ## interpolation
    if dataset_info['is_preproc'] == True:
        interp_order = dataset_info['interp_order']
        print("Interpolating test data...")
        preproc_input(gen_conf, train_conf, is_training=False, interp_order=interp_order) # input pre-processing

    print("Start evaluating ... load data and model...")
    ## load data
    input_data, output_data, filename_list = read_dataset(gen_conf, train_conf, flag)

    # mean, std = read_meanstd(gen_conf, conf, case_name) # for patch
    # mean_out, std_out = read_meanstd_MUDI_output(gen_conf, conf) # for output 4d image
    model = read_model(gen_conf, train_conf, case_name)

    print("Evaluate model ...")
    ## test and save output
    # test_model(gen_conf, test_conf, input_data, model, None, None, mean_out, std_out, case_name)
    eval_model(gen_conf, train_conf, input_data, output_data, filename_list, model, None, None, None, None, flag, case_name)

    return model

# latest one
def eval_model(gen_conf,
               test_conf,
               input_data,
               output_data,
               filename_list,
               model,
               mean,
               std,
               mean_out,
               std_out,
               flag = 'train',
               case_name = 0) :
    num_classes = gen_conf['num_classes']
    output_shape = test_conf['output_shape']
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]
    dimensions = dataset_info['dimensions']
    if flag == 'train':
        num_volumes = dataset_info['num_volumes'][0]
    elif flag == 'eval':
        num_volumes = dataset_info['num_volumes'][1]
    num_modalities = dataset_info['modalities'] # designed number of modalities, a particular setup in test
    real_num_modalities = dataset_info['real_modalities']
    batch_size = test_conf['batch_size']

    ## compute mse
    mse_array = []
    error_degree_label_list = []
    name_list = []
    mse_value = []
    idx_list = []


    for vol_idx in range(0, num_volumes) :
        recon_im_4d = np.zeros(dimensions+(real_num_modalities, ))
        for mod_idx in range(real_num_modalities):
            test_index = vol_idx * real_num_modalities + mod_idx # index of data
            ## should only sample one subject
            if input_data.ndim == 6 :
                input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:2]+input_data.shape[3:])
                output_data_temp = np.reshape(output_data[test_index], (1, )+output_data.shape[1:2]+output_data.shape[3:])
            else:
                input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:])
                output_data_temp = np.reshape(output_data[test_index], (1, )+output_data.shape[1:])

            filename_modality = filename_list[test_index]
            x_test, y_test = overlap_patching_test(gen_conf, test_conf, input_data_temp,
                         output_data_temp,
                         trainTestFlag = 'train',
                         representative_modality = 0)

            # x_test = normalise_volume(x_test, num_modalities, mean['input'], std['input'])
            patch_index = np.arange(len(x_test))
            patch_index_absolute = patch_index + mod_idx*len(x_test)

            for k in patch_index:
                filename_modality_index = filename_modality + str(k)
                name_list.append(filename_modality_index)



            recon_im = model.predict(x_test, verbose=test_conf['verbose'])

            # compute the mse error here and generate the error-level label

            #mse = tf.keras.losses.MeanSquaredError(image_high_re, recon_im).numpy()

            mse = tf.keras.losses.mean_squared_error(y_test.reshape((len(x_test),-1)), recon_im.reshape((len(x_test),-1)))

            #print('@@@@@@@@@@@@@@2', mse.shape)
            error_label_degree = np.zeros((mse.shape))
            mse = mse.numpy()
            error_label_degree[mse>=0] = 0
            error_label_degree[mse>=0.094859] = 1
            error_label_degree[mse>=0.187877] = 2
            error_label_degree[mse>=0.532296] = 3

            error_degree_label_list.extend(error_label_degree)
            mse_value.extend(mse)
            #name_list.extend(mod_idx)
            idx_list.extend(patch_index_absolute)


            # de-normalisation before prediction
            # recon_im = denormalise_volume(recon_im, num_modalities, mean['output'], std['output'])

            #print("Reconstructing ...")
            #recon_im_temp = reconstruct_volume_imaging4(gen_conf, test_conf, recon_im[:, 0])

            # compute MSE
            #print("Compute MSE ...")
            #mask_volume = (output_data[test_index, 0] != 0)
            #mse_mod = np.mean((output_data[test_index, 0] - recon_im_temp) ** 2 * mask_volume)
            #print("The MSE of Subject {}th, Modality {}th is {}".format(vol_idx, mod_idx, mse_mod))
            #mse_array.append(mse_mod)

            # save recon image to 4d structure
            #recon_im_4d[:, :, :, mod_idx] = recon_im_temp
            del x_test, recon_im

        # image-level
        # recon_im_4d = denormalise_4d_volume(recon_im_4d, mean_out, std_out) # cancel denormalise, update [31/08/20]
        #save_volume(gen_conf, test_conf, recon_im_4d, (vol_idx, case_name), flag )

    # save mse_array
    print(np.array(name_list).shape)
    print(np.array(mse_value).shape)
    print(np.array(error_degree_label_list).shape)
    Data4stage2 = pd.DataFrame({'index':idx_list, 'Name':name_list, 'MSE': mse_value, 'Level':error_degree_label_list})
    Data4stage2.to_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_test_label.csv', index = None, encoding='utf8')

    return True

def normalise_volume(input_data, num_modalities, mean, std) :
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)) :
        for modality in range(num_modalities) :
            input_data_tmp[vol_idx, modality] -= mean[modality]
            input_data_tmp[vol_idx, modality] /= std[modality]
    return input_data_tmp

def denormalise_volume(input_data, num_modalities, mean, std) :
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)) :
        for modality in range(num_modalities) :
            input_data_tmp[vol_idx, modality] *= std[modality]
            input_data_tmp[vol_idx, modality] += mean[modality]
    return input_data_tmp

def denormalise_4d_volume(input_data, mean, std) :
    '''
    :param input_data: 4d (x, y, z, t)
    :param mean:
    :param std:
    :return:
    '''
    input_data_tmp = np.copy(input_data)
    num_modalities = input_data_tmp.shape[3] # t
    for modality in range(num_modalities) :
        input_data_tmp[:, :, :, modality] *= std[modality]
        input_data_tmp[:, :, :, modality] += mean[modality]
    return input_data_tmp

def compute_statistics(input_data, num_modalities) :
    mean = np.zeros((num_modalities, ))
    std = np.zeros((num_modalities, ))

    for modality in range(num_modalities) :
        modality_data = input_data[:, modality]
        mean[modality] = np.mean(modality_data)
        std[modality] = np.std(modality_data)

    return mean, std


def generate_output_filename(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension):
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)
