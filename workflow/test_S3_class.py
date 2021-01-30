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


def generate_class(gen_conf, train_conf) :
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_volumes = dataset_info['num_volumes']

    mse_array = []

    ## interpolation
    if dataset_info['is_preproc'] == True:
        print("Interpolating training data...")
        interp_order = dataset_info['interp_order']
        preproc_input(gen_conf, train_conf, is_training=True, interp_order=interp_order) # input pre-processing

    print("Set up data loader ...")
    train_generator, val_generator = DefineTrainValSuperMudiDataloader_Stage3_class(gen_conf, train_conf, para='para')


    for case in range(train_conf['cases']):
        print("Start Case " + str(case) + " training...")
        model = train_model_generator(gen_conf, train_conf, train_generator, val_generator, case)
        #mse_array.append(mse)

    # write to file
    #save_msecorr_array(gen_conf, train_conf, mse_array, None)
    # train_generator.clear_extracted_files() # remove the extracted patch folder
    # update[25/08], the extracted files will be cleared outside the iqt code

    return model

# latest one
def train_model_generator(gen_conf, train_conf, train_generator, val_generator, case_name = 0):
    approach = train_conf['approach']
    dimension = train_conf['dimension']
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_classes = gen_conf['num_classes']
    num_modalities = dataset_info['modalities'] # designed number of modalities, a particular setup in training
    output_shape = train_conf['output_shape'] ## output patch size (32, 32, 32)
    patch_shape = train_conf['patch_shape']  ## input patch size (32, 32, 8)

    First_level = []
    Second_level = []
    Third_level = []
    Fourth_level = []

    First_level_name = [] 
    Second_level_name = []
    Third_level_name = [] 
    Fourth_level_name = []

    First_level_label = []
    Second_level_label = []
    Third_level_label = []
    Fourth_level_label = []

    batch_normalization = True
    # define model
    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']
    lr = train_conf['learning_rate']
    decay = train_conf['decay']

    pre_model = read_model(gen_conf, train_conf, case_name)
    base_model = Model(inputs=pre_model.input, outputs=pre_model.get_layer('batch_normalization_12').output)

    input_2 = Input(shape=(6))
    
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    
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
    final_model = Model(inputs=[base_model.input, input_2], outputs=predictions)
    #final_model = Model(inputs=base_model.input, outputs=predictions)


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
    final_model.load_weights('/cluster/project0/IQT_Nigeria/others/SuperMudi/results/iso_lr1e-3_p888_sbj4_nf64/models/MUDI/S2-para-batchnormalization-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5')


    #print("Set up data loader ...")
    #train_generator, val_generator = DefineTrainValSuperMudiDataloader_new(gen_conf, train_conf)

    ds = iter(train_generator)
    for step in range(len(train_generator)):


        images_low_re, error_label, batch_name = next(ds)

        #print('@@@@@@@@@@@', np.shape(np.array(images_low_re)))
        #recon_im = model.predict(images_low_re)
        #recon_im = model.predict(images_low_re, verbose=train_conf['verbose'])

        predictions = final_model.predict(images_low_re)
        # compute the mse error here and generate the error-level label

        #mse = tf.keras.losses.MeanSquaredError(image_high_re, recon_im).numpy()

        #print('@@@@@@@@@@@@@@2', mse.shape)
        decode_onehot = np.argmax(predictions, axis=1)
        if decode_onehot == 0:
            First_level.extend(decode_onehot)
            First_level_name.extend(batch_name)
            First_level_label.extend(error_label)
        elif decode_onehot == 1:
            Second_level.extend(decode_onehot)
            Second_level_name.extend(batch_name)
            Second_level_label.extend(error_label)
        elif decode_onehot == 2:
            Third_level.extend(decode_onehot)
            Third_level_name.extend(batch_name)
            Third_level_label.extend(error_label)
        elif decode_onehot == 3:
            Fourth_level.extend(decode_onehot)
            Fourth_level_name.extend(batch_name)
            Fourth_level_label.extend(error_label)
        else:
            raise KeyError


        # de-normalisation before prediction
        # recon_im = denormalise_volume(recon_im, num_modalities, mean['output'], std['output'])

        #print("Reconstructing ...")
        #recon_im_4d[:, :, :, mod_idx] = reconstruct_volume_imaging4(gen_conf, test_conf, recon_im[:, 0])
        del images_low_re

        # image-level
        # recon_im_4d = denormalise_4d_volume(recon_im_4d, mean_out, std_out) # cancel denormalise, update [31/08/20]
        #save_volume(gen_conf, test_conf, recon_im_4d, (vol_idx, case_name) )

    # Four separation

    print(np.array(First_level).shape)
    print(np.array(Second_level).shape)
    print(np.array(Third_level).shape)
    print(np.array(Fourth_level).shape)
    Data4stage2 = pd.DataFrame({'Name':First_level_name, 'Label': First_level_label, 'Prediction':First_level})
    Data4stage2.to_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_batchnormalization_Level_1_label.csv', index = None, encoding='utf8')

    Data4stage2 = pd.DataFrame({'Name':Second_level_name, 'Label': Second_level_label, 'Prediction':Second_level})
    Data4stage2.to_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_batchnormalization_Level_2_label.csv', index = None, encoding='utf8')

    Data4stage2 = pd.DataFrame({'Name':Third_level_name, 'Label': Third_level_label, 'Prediction':Third_level})
    Data4stage2.to_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_batchnormalization_Level_3_label.csv', index = None, encoding='utf8')

    Data4stage2 = pd.DataFrame({'Name':Fourth_level_name, 'Label': Fourth_level_label, 'Prediction':Fourth_level})
    Data4stage2.to_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_batchnormalization_Level_4_label.csv', index = None, encoding='utf8')

    return final_model



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
