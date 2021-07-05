import numpy as np
import os
import tensorflow as tf
import pandas as pd
from architectures.arch_creator import generate_model, generate_model_s3
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


def test_S3_test(gen_conf_train, gen_conf_test, train_conf_1, test_conf, flag = 'train', case_name = 0) :
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
    if batch_normalization==True:
        x = BatchNormalization(axis=1)(x)
        x = Dense(128, activation='relu', name='global_dense2')(x)
    if batch_normalization==True:
        x = BatchNormalization(axis=1)(x)
    #x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='global_dense3')(x)
    if batch_normalization==True:
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
    stored_path = '{}/{}/S2-para-batchnormalization-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5'.format(gen_conf['model_path'],train_conf_1['dataset'])
    final_model.load_weights(stored_path)

    print("Evaluate model ...")
    eval_model(gen_conf, gen_conf_train, train_conf_1, train_conf, input_data, output_data, filename_list, final_model, para, None, None, None, None, flag, case_name)

    return final_model

# latest one
def eval_model(gen_conf,
                gen_conf_train,
                train_conf_1,
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

    case_name = 'para-batchnormalization'
    ## compute mse
    name_list = []
    recon_im_list = []
    label_level_modality_cat_list = []
    prediction_list = []
    mse_list = []
    mse_array = []

    model_1 = generate_model_s3(gen_conf_train, train_conf_1)
    model_2 = generate_model_s3(gen_conf_train, train_conf_1)
    model_3 = generate_model_s3(gen_conf_train, train_conf_1)
    model_4 = generate_model_s3(gen_conf_train, train_conf_1)
    
    stored_path_1 = '{}/{}/S3_para_batchnormalization_1-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5'.format(gen_conf['model_path'],train_conf_1['dataset'])
    stored_path_2 = '{}/{}/S3_para_batchnormalization_2-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5'.format(gen_conf['model_path'],train_conf_1['dataset'])
    stored_path_3 = '{}/{}/S3_para_batchnormalization_3-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5'.format(gen_conf['model_path'],train_conf_1['dataset'])
    stored_path_4 = '{}/{}/S3_para_batchnormalization_4-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5'.format(gen_conf['model_path'],train_conf_1['dataset'])

    model_1.load_weights()
    model_2.load_weights()
    model_3.load_weights()
    model_4.load_weights()

    para_file=open('./csv_files/parameters.txt')
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
            x_test, _ = overlap_patching(gen_conf, test_conf, input_data_temp,
                         output_data = None,
                         trainTestFlag = 'test',
                         representative_modality = 0)

            # x_test = normalise_volume(x_test, num_modalities, mean['input'], std['input'])

            x_test_ensemble = np.zeros((x_test.shape[0:-3]+(16,16,16)))

            patch_index = np.arange(len(x_test))
            patch_index_absolute = patch_index + mod_idx*len(x_test)

            #label_level_modality = label_level_list_in[patch_index_absolute]

            for k in patch_index:
                filename_modality_index = filename_modality + '-' + str(k)
                name_list.append(filename_modality_index)

            if para=='para':
                para_modality = para_numpy[mod_idx]
                para_modality_expand = np.tile(para_modality,(1056,1))
                x_test_en = [x_test, para_modality_expand]

            print('x_test shape is: ', np.shape(x_test))
            print('para_modality_expand shape is: ', np.shape(para_modality_expand))

            recon_im = model.predict(x_test_en, verbose=test_conf['verbose'])


            recon_im_decode = np.argmax(recon_im, axis=1)

            index_1=[i for i,x in enumerate(recon_im_decode) if x==0]
            index_2=[i for i,x in enumerate(recon_im_decode) if x==1]
            index_3=[i for i,x in enumerate(recon_im_decode) if x==2]
            index_4=[i for i,x in enumerate(recon_im_decode) if x==3]

            
            assert (len(index_1)+len(index_2)+len(index_3)+len(index_4))==1056

            if len(index_1)!= 0:
                data_for_decoder_1 = x_test[index_1]
                recon_im_1 = model_1.predict(data_for_decoder_1, verbose=test_conf['verbose'])
                x_test_ensemble[index_1] = recon_im_1
            if len(index_2)!= 0:
                data_for_decoder_2 = x_test[index_2]
                recon_im_2 = model_2.predict(data_for_decoder_2, verbose=test_conf['verbose'])
                x_test_ensemble[index_2] = recon_im_2
            if len(index_3)!= 0:
                data_for_decoder_3 = x_test[index_3]
                recon_im_3 = model_3.predict(data_for_decoder_3, verbose=test_conf['verbose'])
                x_test_ensemble[index_3] = recon_im_3
            if len(index_4)!= 0:
                data_for_decoder_4 = x_test[index_4]
                recon_im_4 = model_4.predict(data_for_decoder_4, verbose=test_conf['verbose'])
                x_test_ensemble[index_4] = recon_im_4


            print("Reconstructing ...")
            #print(np.shape(x_test_ensemble[:, 0]))
            recon_im_temp = reconstruct_volume_imaging4(gen_conf, test_conf, x_test_ensemble[:, 0])
            # compute MSE
            print("Compute MSE ...")
            mask_volume = (output_data[test_index, 0] != 0)
            mse_mod = np.mean((output_data[test_index, 0] - recon_im_temp) ** 2 * mask_volume)
            print("The MSE of Subject {}th, Modality {}th is {}".format(vol_idx, mod_idx, mse_mod))
            mse_array.append(mse_mod)

            # save recon image to 4d structure
            recon_im_4d[:, :, :, mod_idx] = recon_im_temp
            print('4D shape: ',np.shape(recon_im_4d))
            del x_test, x_test_ensemble

        # image-level
        # recon_im_4d = denormalise_4d_volume(recon_im_4d, mean_out, std_out) # cancel denormalise, update [31/08/20]
        save_volume(gen_conf, test_conf, recon_im_4d, (vol_idx, case_name), flag )

    # save mse_array
    txt_file = generate_output_filename(
        gen_conf['evaluation_path'],
        test_conf['dataset'],
        flag+'-c'+str(case_name),
        test_conf['approach'],
        test_conf['dimension'],
        str(test_conf['patch_shape']),
        str(test_conf['extraction_step']),
        'txt')
    if not os.path.isdir(os.path.dirname(txt_file)):
        os.makedirs(os.path.dirname(txt_file))
    np.savetxt(txt_file, mse_array)

    mse = np.mean(np.array(mse_array))
    print("The average of MSE over all subjects and modalities is {}".format(mse))


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
