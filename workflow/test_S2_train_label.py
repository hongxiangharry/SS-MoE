import numpy as np
import tensorflow as tf
import pandas as pd
from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks
from utils.ioutils import read_dataset, save_volume, read_model, generate_output_filename, read_meanstd, read_meanstd_MUDI_output, save_msecorr_array
from utils.reconstruction import reconstruct_volume_imaging, reconstruct_volume_imaging4, reconstruct_volume_imaging3
from utils.patching_utils import overlap_patching
from utils.preprocessing_util import preproc_input
from utils.visualize_model import visualize_model
import matplotlib.pyplot as plt
import sys
from dataloader.SuperMUDI import DefineTrainValSuperMudiDataloader_new
from .train import *



def generate_train_label(gen_conf, train_conf) :
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_volumes = dataset_info['num_volumes']

    mse_array = []

    ## interpolation
    if dataset_info['is_preproc'] == True:
        print("Interpolating training data...")
        interp_order = dataset_info['interp_order']
        preproc_input(gen_conf, train_conf, is_training=True, interp_order=interp_order) # input pre-processing

    print("Set up data loader ...")
    train_generator, val_generator = DefineTrainValSuperMudiDataloader_new(gen_conf, train_conf)


    for case in range(train_conf['cases']):
        print("Start Case " + str(case) + " training...")
        model, mse = train_model_generator(gen_conf, train_conf, train_generator, val_generator, case)
        mse_array.append(mse)

    # write to file
    save_msecorr_array(gen_conf, train_conf, mse_array, None)
    # train_generator.clear_extracted_files() # remove the extracted patch folder
    # update[25/08], the extracted files will be cleared outside the iqt code

    return model, mse_array

# latest one
def train_model_generator(gen_conf, train_conf, train_generator, val_generator, case_name = 0):
    approach = train_conf['approach']
    dimension = train_conf['dimension']
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_classes = gen_conf['num_classes']
    num_modalities = dataset_info['modalities'] # designed number of modalities, a particular setup in training
    output_shape = train_conf['output_shape'] ## output patch size (32, 32, 32)
    patch_shape = train_conf['patch_shape']  ## input patch size (32, 32, 8)

    error_degree_label_list = []
    name_list = []
    mse_value = []
    idx_list = []

    model = read_model(gen_conf, train_conf, case_name)

    #print("Set up data loader ...")
    #train_generator, val_generator = DefineTrainValSuperMudiDataloader_new(gen_conf, train_conf)

    ds = iter(train_generator)
    for step in range(len(train_generator)-1):


        images_low_re, image_high_re, filename, idx = next(ds)
        #recon_im = model.predict(images_low_re)
        #recon_im = model.predict(images_low_re, verbose=train_conf['verbose'])
        recon_im = model.predict(images_low_re)
        # compute the mse error here and generate the error-level label

        #mse = tf.keras.losses.MeanSquaredError(image_high_re, recon_im).numpy()

        mse = tf.keras.losses.mean_squared_error(image_high_re.reshape((64,-1)), recon_im.reshape((64,-1)))

        #print('@@@@@@@@@@@@@@2', mse.shape)
        error_label_degree = np.zeros((mse.shape))
        mse = mse.numpy()
        error_label_degree[mse>=0] = 0
        error_label_degree[mse>=1] = 1
        error_label_degree[mse>=10] = 2
        error_label_degree[mse>=100] = 3

        error_degree_label_list.extend(error_label_degree)
        mse_value.extend(mse)
        name_list.extend(filename)
        idx_list.extend(idx)
        # de-normalisation before prediction
        # recon_im = denormalise_volume(recon_im, num_modalities, mean['output'], std['output'])

        #print("Reconstructing ...")
        #recon_im_4d[:, :, :, mod_idx] = reconstruct_volume_imaging4(gen_conf, test_conf, recon_im[:, 0])
        del images_low_re, recon_im

        # image-level
        # recon_im_4d = denormalise_4d_volume(recon_im_4d, mean_out, std_out) # cancel denormalise, update [31/08/20]
        #save_volume(gen_conf, test_conf, recon_im_4d, (vol_idx, case_name) )

    # Four separation



    print(np.array(name_list).shape)
    print(np.array(mse_value).shape)
    print(np.array(error_degree_label_list).shape)
    Data4stage2 = pd.DataFrame({'index':idx_list, 'Name':name_list, 'MSE': mse_value, 'Level':error_degree_label_list})
    Data4stage2.to_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_label.csv', index = None, encoding='utf8')

    return model, mse



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
