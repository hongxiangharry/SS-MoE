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
from dataloader.SuperMUDI import DefineTrainValSuperMudiDataloader_Stage3_class, DefineTrainValSuperMudiDataloader_Stage2
from .train import *
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, GlobalAveragePooling3D # return the (batchsize, channel)
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Dense
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, Dropout
from tensorflow.keras.layers import Permute, Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add as layer_add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from utils.loss_utils import L2TV, L2L2
from tensorflow.keras.utils import to_categorical




def test_S2_test(gen_conf_train, gen_conf_test, train_conf_1, test_conf, flag = 'train', case_name = 0) :
    '''
    :param gen_conf:
    :param train_conf:
    :param flag: train/eval
    :param case_name:
    :return:
    '''
    gen_conf = gen_conf_test
    train_conf = test_conf

    para = 'para'
    batch_normalization = True

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
    
    
    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']
    lr = train_conf['learning_rate']
    decay = train_conf['decay']
    gen_conf_train['model_path'] = os.path.join(gen_conf_train['base_path'], 'iso_lr1e-3_p888_sbj4_nf64', gen_conf_train['model_path_'])
    print(gen_conf_train['model_path'])
    pre_model = read_model(gen_conf_train, train_conf_1, case_name)
    base_model = Model(inputs=pre_model.input, outputs=pre_model.get_layer('batch_normalization_12').output)

    input_2 = Input(shape=(6))
    
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    
    if para=='para':
        x = concatenate([x, input_2],axis=1)
    x = Dense(256, activation='relu', name='global_dense1')(x)
    if batch_normalization:
        x = BatchNormalization(axis=1)(x)
    x = Dense(128, activation='relu', name='global_dense2')(x)
    if batch_normalization:
        x = BatchNormalization(axis=1)(x)
    #x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='global_dense3')(x)
    if batch_normalization:
        x = BatchNormalization(axis=1)(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax', name='global_predictions')(x)

    if para=='para':
        final_model = Model(inputs=[base_model.input, input_2], outputs=predictions)
    else:
        final_model = Model(inputs=base_model.input, outputs=predictions)


    for layer in base_model.layers:
        layer.trainable = False

    # optimizer
    if optimizer == 'Adam' :
        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
    elif optimizer == 'SGD' :
        optimizer =  SGD(lr=lr, nesterov=True)

    # loss
    loss = tf.keras.losses.CategoricalCrossentropy()

    final_model.compile(loss=loss, optimizer=optimizer, metrics='accuracy')

    # choose weight here
    if para=='para':
        final_model.load_weights('/cluster/project0/IQT_Nigeria/others/SuperMudi/results/iso_lr1e-3_p888_sbj4_nf64/models/MUDI/S2-para-batchnormalization-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5')
    else:
        final_model.load_weights('/cluster/project0/IQT_Nigeria/others/SuperMudi/results/iso_lr1e-3_p888_sbj4_nf64/models/MUDI/S2-nopara-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5')

    #final_model = tf.keras.models.load_model('/cluster/project0/IQT_Nigeria/others/SuperMudi/results/iso_lr1e-3_p888_sbj4_nf64/models/MUDI/S2_wholemodel-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5')
    print("Evaluate model ...")
    ## test and save output
    # test_model(gen_conf, test_conf, input_data, model, None, None, mean_out, std_out, case_name)

    

    eval_model(gen_conf, train_conf, input_data, output_data, filename_list, final_model, para, None, None, None, None, flag, case_name)

    return final_model

# latest one
def eval_model(gen_conf,
               test_conf,
               input_data,
               output_data,
               filename_list,
               model,
               para,
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
    name_list = []
    recon_im_list = []
    label_level_modality_cat_list = []
    prediction_list = []
    idx_list = []

    data_list = pd.read_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_test_label.csv')
    label_in = data_list[['index', 'Level']]
    index_in = np.array(label_in['index'])
    label_level_list_in = np.array(label_in['Level'])


    para_file=open('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/parameters.txt')
    txt=para_file.readlines()
    para_list=[]
    for w in txt:
        #w=w.replace('\n','')
        w=w.split()
        w = list(map(float, w))
        para_list.append(w)

    para_numpy = np.array(para_list)

    d_mean = np.mean(para_numpy, axis=0)
    d_std = np.std(para_numpy,axis=0)

    para_numpy = (para_numpy-d_mean)/d_std

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

            label_level_modality = label_level_list_in[patch_index_absolute]

            for k in patch_index:
                filename_modality_index = filename_modality + str(k)
                name_list.append(filename_modality_index)


            if para=='para':
                para_modality = para_numpy[mod_idx]
                para_modality_expand = np.tile(para_modality,(183,1))

                x_test = [x_test, para_modality_expand]

            recon_im = model.predict(x_test, verbose=test_conf['verbose'])

            label_level_modality_cat = to_categorical(np.array(label_level_modality), 4)

            recon_im_decode = np.argmax(recon_im, axis=1)

            # compute the mse error here and generate the error-level label

            #mse = tf.keras.losses.MeanSquaredError(image_high_re, recon_im).numpy()

            #print(recon_im)
            #loss = tf.keras.losses.CategoricalCrossentropy()(label_level_modality_cat, recon_im).numpy()

            #accuracy = tf.keras.metrics.CategoricalAccuracy()(label_level_modality_cat, recon_im).numpy()



            #print('@@@@@@@@@@@@@@2', mse.shape)

            recon_im_list.extend(recon_im)
            label_level_modality_cat_list.extend(label_level_modality_cat)
            #name_list.extend(mod_idx)
            prediction_list.extend(recon_im_decode)


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

    loss = tf.keras.losses.CategoricalCrossentropy()(label_level_modality_cat_list, recon_im_list).numpy()

    accuracy = tf.keras.metrics.CategoricalAccuracy()(label_level_modality_cat_list, recon_im_list).numpy()
    # save mse_array
    print(np.array(prediction_list).mean(axis=0))
    print(np.array(label_level_modality_cat_list).mean(axis=0))
    print(np.array(recon_im_list).shape)

    print('loss is: ',loss)
    print('accuracy is: ', accuracy)
    Data4stage2 = pd.DataFrame({'Name':name_list, 'Label':label_level_list_in, 'Logit':recon_im_list, 'Prediction':prediction_list})
    Data4stage2.to_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_withpara_batchnormalization_test_results_loss_accuracy.csv', index = None, encoding='utf8')

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
