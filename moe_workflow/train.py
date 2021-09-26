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

import tensorflow as tf
import numpy as np
from architectures.arch_creator import generate_model, generate_model_s3
from utils.callbacks import generate_callbacks, generate_callbacks_S2, generate_callbacks_S3
from tensorflow.keras import callbacks
from utils.ioutils import read_dataset, read_model, save_meanstd, read_meanstd, save_msecorr_array
from utils.patching_utils import overlap_patching
from utils.visualize_model import visualize_model
import matplotlib.pyplot as plt
from utils.preprocessing_util import preproc_input
import sys
from dataloader.SuperMUDI import DefineTrainValSuperMudiDataloader,DefineTrainValSuperMudiDataloader_moe, DefineTrainValSuperMudiDataloader_Stage2, DefineTrainValSuperMudiDataloader_Stage3
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



ITERATION = 1000
mse_loss = tf.keras.losses.MeanSquaredError()
class_loss_metrics = tf.metrics.Mean(name='class_class')



##############################
################################ stage 1 pretraining



## evaluate_using_training_testing_split
def training(gen_conf, train_conf) :
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_volumes = dataset_info['num_volumes']
    para = 'nopara'
    csv_path = '/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_label_train.csv'
    mse_array = []

    ## interpolation
    if dataset_info['is_preproc'] == True:
        print("Interpolating training data...")
        interp_order = dataset_info['interp_order']
        preproc_input(gen_conf, train_conf, is_training=True, interp_order=interp_order) # input pre-processing

    print("Set up data loader ...")
    train_generator, val_generator = DefineTrainValSuperMudiDataloader(gen_conf, train_conf, para, csv_path)

    x, y = train_generator[0]
    print(x.shape, y.shape)

    for case in range(train_conf['cases']):
        print("Start Case " + str(case) + " training...")
        model, mse = train_model_generator(gen_conf, train_conf, train_generator, val_generator, case)
        mse_array.append(mse)

    # write to file
    save_msecorr_array(gen_conf, train_conf, mse_array, None)
    # train_generator.clear_extracted_files() # remove the extracted patch folder
    # update[25/08], the extracted files will be cleared outside the iqt code

    return mse_array

########################## Hard Mixture of Experts ##########################
def train_moe(gen_conf, train_conf):
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_volumes = dataset_info['num_volumes']
    mse_array = []

    ## interpolation
    if dataset_info['is_preproc'] == True:
        print("Interpolating training data...")
        interp_order = dataset_info['interp_order']
        preproc_input(gen_conf, train_conf, is_training=True, interp_order=interp_order) # input pre-processing

    print("Set up data loader ...")
    train_generator, val_generator = DefineTrainValSuperMudiDataloader_moe(gen_conf, train_conf, is_shuffle_trainval=True)

    x, y = train_generator[0]
    print(x.shape, y.shape)

    for case in range(train_conf['cases']):
        print("Start Case " + str(case) + " training...")
        model, mse = train_model_generator(gen_conf, train_conf, train_generator, val_generator, case)
        mse_array.append(mse)

    # write to file
    save_msecorr_array(gen_conf, train_conf, mse_array, None)
    # train_generator.clear_extracted_files() # remove the extracted patch folder
    # update[25/08], the extracted files will be cleared outside the iqt code

    return mse_array

def train_model_generator(gen_conf, train_conf, train_generator, val_generator, case_name = 0):
    approach = train_conf['approach']
    dimension = train_conf['dimension']
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_classes = gen_conf['num_classes']
    num_modalities = dataset_info['modalities'] # designed number of modalities, a particular setup in training
    output_shape = train_conf['output_shape'] ## output patch size (32, 32, 32)
    patch_shape = train_conf['patch_shape']  ## input patch size (32, 32, 8)

    if train_conf['num_epochs'] != 0:
        # training callbacks
        callbacks = generate_callbacks(gen_conf, train_conf, case_name)
        if callbacks is not None: # train or not ?
            model = __train_model(gen_conf, train_conf, train_generator, val_generator, case_name, callbacks)
        else:
            model = read_model(gen_conf, train_conf, case_name)

        if val_generator is not None:
            ## compute mse and corr on validation set
            mse = model.evaluate(val_generator,
                                 batch_size=train_conf['batch_size'],
                                 workers=train_conf['workers'],
                                 use_multiprocessing=train_conf['use_multiprocessing'],
                                 verbose=train_conf['verbose'])
        else:
            mse = model.evaluate(train_generator,
                                 batch_size=train_conf['batch_size'],
                                 workers=train_conf['workers'],
                                 use_multiprocessing=train_conf['use_multiprocessing'],
                                 verbose=train_conf['verbose'])
    else:
        model = []
        mse = None

    return model, mse

def __train_model(gen_conf, train_conf, train_generator, val_generator, case_name, callbacks, vis_flag = False):
    model = generate_model(gen_conf, train_conf)

    print(model.summary()) # print model summary

    history = model.fit(
        train_generator,
        batch_size=train_conf['batch_size'],
        epochs=train_conf['num_epochs'],
        validation_data=val_generator,
        shuffle=train_conf['shuffle'],
        verbose=train_conf['verbose'],
        workers=train_conf['workers'],
        use_multiprocessing=train_conf['use_multiprocessing'],
        callbacks=callbacks)
    print(history.params) # print model parameters

    # if vis_flag == True:
    visualize_model(model, history, gen_conf, train_conf, case_name, vis_flag)

    return model




########################## 20210120
########################## stage 2.1

def training_S2(gen_conf, train_conf) :
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_volumes = dataset_info['num_volumes']

    mse_array = []

    ## interpolation
    if dataset_info['is_preproc'] == True:
        print("Interpolating training data...")
        interp_order = dataset_info['interp_order']
        preproc_input(gen_conf, train_conf, is_training=True, interp_order=interp_order) # input pre-processing

    print("Set up data loader ...")
    train_generator, val_generator = DefineTrainValSuperMudiDataloader_Stage2(gen_conf, train_conf, Para='para')

    for case in range(train_conf['cases']):
        print("Start Case " + str(case) + " training...")
        model, mse = train_model_generator_S2(gen_conf, train_conf, train_generator, val_generator, case)
        mse_array.append(mse)

    # write to file
    #save_msecorr_array_S2(gen_conf, train_conf, mse_array, None)
    # train_generator.clear_extracted_files() # remove the extracted patch folder
    # update[25/08], the extracted files will be cleared outside the iqt code

    return model, mse_array

def train_model_generator_S2(gen_conf, train_conf, train_generator, val_generator, case_name = 0):
    approach = train_conf['approach']
    dimension = train_conf['dimension']
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_classes = gen_conf['num_classes']
    num_modalities = dataset_info['modalities'] # designed number of modalities, a particular setup in training
    output_shape = train_conf['output_shape'] ## output patch size (32, 32, 32)
    patch_shape = train_conf['patch_shape']  ## input patch size (32, 32, 8)

    if train_conf['num_epochs'] != 0:
        # training callbacks
        callbacks = generate_callbacks_S2(gen_conf, train_conf, case_name)
        if callbacks is not None: # train or not ?
            model = __train_model_S2(gen_conf, train_conf, train_generator, val_generator, case_name, callbacks)
        else:
            model = read_model(gen_conf, train_conf, case_name)

        if val_generator is not None:
            ## compute mse and corr on validation set
            mse = model.evaluate(val_generator,
                                 batch_size=train_conf['batch_size'],
                                 workers=train_conf['workers'],
                                 use_multiprocessing=train_conf['use_multiprocessing'],
                                 verbose=train_conf['verbose'])
        else:
            mse = model.evaluate(train_generator,
                                 batch_size=train_conf['batch_size'],
                                 workers=train_conf['workers'],
                                 use_multiprocessing=train_conf['use_multiprocessing'],
                                 verbose=train_conf['verbose'])
    else:
        model = []
        mse = None

    return model, mse

def __train_model_S2(gen_conf, train_conf, train_generator, val_generator, case_name, callbacks_1, vis_flag=False):
    
    #model = generate_model(gen_conf, train_conf)
    loss = train_conf['loss']
    metrics = train_conf['metrics']
    optimizer = train_conf['optimizer']
    lr = train_conf['learning_rate']
    decay = train_conf['decay']
    batch_normalization = True

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
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    elif optimizer == 'SGD' :
        optimizer =  SGD(lr=lr, nesterov=True)

    # loss
    loss = tf.keras.losses.CategoricalCrossentropy()

    final_model.compile(loss=loss, optimizer=optimizer, metrics='accuracy')


    
    callbacks_S2 = generate_callbacks_S2(gen_conf, train_conf, case_name='S2-para-batchnormalization')


    print(final_model.summary()) # print model summary

    history = final_model.fit(
        train_generator,
        batch_size=train_conf['batch_size'],
        epochs=30,
        validation_data=val_generator,
        shuffle=train_conf['shuffle'],
        verbose=train_conf['verbose'],
        workers=train_conf['workers'],
        use_multiprocessing=train_conf['use_multiprocessing'],
        callbacks=callbacks_S2)

    print(history.params) # print model parameters

    # if vis_flag == True:
    visualize_model(final_model, history, gen_conf, train_conf, case_name, vis_flag)

    return final_model



########################################### 20210120
########################################## stage 2.2



def training_S3(gen_conf, train_conf) :
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_volumes = dataset_info['num_volumes']

    mse_array = []

    ## interpolation
    if dataset_info['is_preproc'] == True:
        print("Interpolating training data...")
        interp_order = dataset_info['interp_order']
        preproc_input(gen_conf, train_conf, is_training=True, interp_order=interp_order) # input pre-processing

    print("Set up data loader ...")
    train_generator_1, val_generator_1 = DefineTrainValSuperMudiDataloader_Stage3(gen_conf, train_conf, csv_path='/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_batchnormalization_Level_1_label.csv', para='para')
    train_generator_2, val_generator_2 = DefineTrainValSuperMudiDataloader_Stage3(gen_conf, train_conf, csv_path='/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_batchnormalization_Level_2_label.csv', para='para')
    train_generator_3, val_generator_3 = DefineTrainValSuperMudiDataloader_Stage3(gen_conf, train_conf, csv_path='/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_batchnormalization_Level_3_label.csv', para='para')
    train_generator_4, val_generator_4 = DefineTrainValSuperMudiDataloader_Stage3(gen_conf, train_conf, csv_path='/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_batchnormalization_Level_4_label.csv', para='para')
    
    #x, y = train_generator[0]
    #print(x.shape, y.shape)

    for case in range(train_conf['cases']):
        print("Start Case " + str(case) + " training...")
        model_1, model_2, model_3, model_4, mse_1, mse_2, mse_3, mse_4 = train_model_generator_S3(gen_conf, train_conf, train_generator_1, train_generator_2, train_generator_3, train_generator_4, val_generator_1, val_generator_2, val_generator_3, val_generator_4, case)
        mse_array.append(mse_2) # HL comments: bug, why only save mse_2?

    # write to file
    save_msecorr_array(gen_conf, train_conf, mse_array, None)
    # train_generator.clear_extracted_files() # remove the extracted patch folder
    # update[25/08], the extracted files will be cleared outside the iqt code

    return mse_array

def train_model_generator_S3(gen_conf, train_conf, train_generator_1, train_generator_2, train_generator_3, train_generator_4, val_generator_1, val_generator_2, val_generator_3, val_generator_4, case_name = 0):
    approach = train_conf['approach']
    dimension = train_conf['dimension']
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_classes = gen_conf['num_classes']
    num_modalities = dataset_info['modalities'] # designed number of modalities, a particular setup in training
    output_shape = train_conf['output_shape'] ## output patch size (32, 32, 32)
    patch_shape = train_conf['patch_shape']  ## input patch size (32, 32, 8)

    if train_conf['num_epochs'] != 0:
        # training callbacks
        callbacks_1 = generate_callbacks_S3(gen_conf, train_conf, case_name='S3_para_batchnormalization_1')
        callbacks_2 = generate_callbacks_S3(gen_conf, train_conf, case_name='S3_para_batchnormalization_2')
        callbacks_3 = generate_callbacks_S3(gen_conf, train_conf, case_name='S3_para_batchnormalization_3')
        callbacks_4 = generate_callbacks_S3(gen_conf, train_conf, case_name='S3_para_batchnormalization_4')

        if callbacks_1 is not None: # train or not ?
            model_1, model_2, model_3, model_4 = __train_model_S3(gen_conf, train_conf, train_generator_1, train_generator_2, train_generator_3, train_generator_4, val_generator_1, val_generator_2, val_generator_3, val_generator_4, case_name, callbacks_1, callbacks_2,callbacks_3,callbacks_4)
        else:
            model_1, model_2, model_3, model_4 = read_model(gen_conf, train_conf, case_name)

        if val_generator_1 is not None:
            ## compute mse and corr on validation set
            
            mse_1 = model_1.evaluate(val_generator_1,
                                 batch_size=train_conf['batch_size'],
                                 workers=train_conf['workers'],
                                 use_multiprocessing=train_conf['use_multiprocessing'],
                                 verbose=train_conf['verbose'])
            
            mse_2 = model_2.evaluate(val_generator_2,
                                 batch_size=train_conf['batch_size'],
                                 workers=train_conf['workers'],
                                 use_multiprocessing=train_conf['use_multiprocessing'],
                                 verbose=train_conf['verbose'])
            mse_3 = model_3.evaluate(val_generator_3,
                                 batch_size=train_conf['batch_size'],
                                 workers=train_conf['workers'],
                                 use_multiprocessing=train_conf['use_multiprocessing'],
                                 verbose=train_conf['verbose'])
	    
            mse_4 = model_4.evaluate(val_generator_4,
                                 batch_size=train_conf['batch_size'],
                                 workers=train_conf['workers'],
                                 use_multiprocessing=train_conf['use_multiprocessing'],
                                 verbose=train_conf['verbose'])
    else:
        model = []
        mse = None

    return model_1, model_2, model_3, model_4, mse_1, mse_2, mse_3, mse_4

def __train_model_S3(gen_conf, train_conf, train_generator_1, train_generator_2, train_generator_3, train_generator_4, val_generator_1, val_generator_2, val_generator_3, val_generator_4, case_name, callbacks_1, callbacks_2,callbacks_3,callbacks_4, vis_flag=False):
    #model_1 = Model_S3(gen_conf, train_conf)
    #model_2 = Model_S3(gen_conf, train_conf)
    #model_3 = Model_S3(gen_conf, train_conf)
    #model_4 = Model_S3(gen_conf, train_conf)

    model_1 = generate_model_s3(gen_conf, train_conf)
    model_2 = generate_model_s3(gen_conf, train_conf)
    model_3 = generate_model_s3(gen_conf, train_conf)
    model_4 = generate_model_s3(gen_conf, train_conf)

    model_1.load_weights('/cluster/project0/IQT_Nigeria/others/SuperMudi/results/iso_lr1e-3_p888_sbj4_nf64/models/MUDI/0-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5')
    model_2.load_weights('/cluster/project0/IQT_Nigeria/others/SuperMudi/results/iso_lr1e-3_p888_sbj4_nf64/models/MUDI/0-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5')
    model_3.load_weights('/cluster/project0/IQT_Nigeria/others/SuperMudi/results/iso_lr1e-3_p888_sbj4_nf64/models/MUDI/0-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5')
    model_4.load_weights('/cluster/project0/IQT_Nigeria/others/SuperMudi/results/iso_lr1e-3_p888_sbj4_nf64/models/MUDI/0-IsoSRUnet-3-(8, 8, 8)-(4, 4, 4).h5')



    print(model_1.summary()) # print model summary

        
    history = model_1.fit(
        train_generator_1,
        batch_size=train_conf['batch_size'],
        epochs=100,
        validation_data=val_generator_1,
        shuffle=train_conf['shuffle'],
        verbose=train_conf['verbose'],
        workers=train_conf['workers'],
        use_multiprocessing=train_conf['use_multiprocessing'],
        callbacks=callbacks_1)
    print(history.params) # print model parameters
    
    # if vis_flag == True:
    visualize_model(model_1, history, gen_conf, train_conf, case_name, vis_flag)
    

    history = model_2.fit(
        train_generator_2,
        batch_size=train_conf['batch_size'],
        epochs=100,
        validation_data=val_generator_2,
        shuffle=train_conf['shuffle'],
        verbose=train_conf['verbose'],
        workers=train_conf['workers'],
        use_multiprocessing=train_conf['use_multiprocessing'],
        callbacks=callbacks_2)
    print(history.params) # print model parameters


    history = model_3.fit(
        train_generator_3,
        batch_size=train_conf['batch_size'],
        epochs=100,
        validation_data=val_generator_3,
        shuffle=train_conf['shuffle'],
        verbose=train_conf['verbose'],
        workers=train_conf['workers'],
        use_multiprocessing=train_conf['use_multiprocessing'],
        callbacks=callbacks_3)
    print(history.params) # print model parameters

    
    history = model_4.fit(
        train_generator_4,
        batch_size=train_conf['batch_size'],
        epochs=100,
        validation_data=val_generator_4,
        shuffle=train_conf['shuffle'],
        verbose=train_conf['verbose'],
        workers=train_conf['workers'],
        use_multiprocessing=train_conf['use_multiprocessing'],
        callbacks=callbacks_4)
    print(history.params) # print model parameters


    return model_1, model_2, model_3, model_4
    #return  model_2, model_3, model_4



##############################################################








def split_train_val(train_indexes, validation_split) :
    N = len(train_indexes)
    val_volumes = np.int32(np.ceil(N * validation_split))
    train_volumes = N - val_volumes

    return train_indexes[:train_volumes], train_indexes[train_volumes:]

def compute_statistics(input_data, num_modalities) :
    mean = np.zeros((num_modalities, ))
    std = np.zeros((num_modalities, ))

    for modality in range(num_modalities) :
        modality_data = input_data[:, modality]
        mean[modality] = np.mean(modality_data)
        std[modality] = np.std(modality_data)

    return mean, std

def normalise_set(input_data, num_modalities, mean, std) :
    input_data_tmp = np.copy(input_data)
    for vol_idx in range(len(input_data_tmp)) :
        for modality in range(num_modalities) :
            input_data_tmp[vol_idx, modality] -= mean[modality]
            input_data_tmp[vol_idx, modality] /= std[modality]
    return input_data_tmp

# def read_model(gen_conf, train_conf, case_name) :
#     model = generate_model(gen_conf, train_conf)

#     model_filename = generate_output_filename(
#         gen_conf['model_path'],
#         train_conf['dataset'],
#         case_name,
#         train_conf['approach'],
#         train_conf['dimension'],
#         str(train_conf['patch_shape']),
#         str(train_conf['extraction_step']),
#         'h5')

#     model.load_weights(model_filename)

#     return model

def debug_plot_patch(input_patch, output_patch):

    fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(9.3, 4),
                            subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)

    for idx, ax in enumerate(axs.flat[:6]):
        ax.imshow(input_patch[idx+1000, 0, 16, :, :], interpolation=None, cmap='gray', aspect=0.25)
        # ax.set_title(str(interp_method))
    for idx, ax in enumerate(axs.flat[6:]):
        ax.imshow(output_patch[idx+1000, 0, 16, :, :], interpolation=None, cmap='gray')

    plt.tight_layout()
    plt.show()
