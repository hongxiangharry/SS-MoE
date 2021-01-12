import numpy as np
from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks
from utils.ioutils import read_dataset, read_model, save_meanstd, read_meanstd, save_msecorr_array
from utils.patching_utils import overlap_patching
from utils.visualize_model import visualize_model
import matplotlib.pyplot as plt
from utils.preprocessing_util import preproc_input

## evaluate_using_training_testing_split
def training(gen_conf, train_conf) :
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_modalities = dataset_info['modalities']
    num_volumes = dataset_info['num_volumes']
    dimension = train_conf['dimension']
    patch_shape = train_conf['patch_shape']
    output_shape = train_conf['output_shape']

    mean_array = []
    std_array = []
    mse_array = []
    corr_array = []

    ## interpolation
    if dataset_info['is_preproc'] == True:
        print("Interpolating training data...")
        interp_order = dataset_info['interp_order']
        preproc_input(gen_conf, train_conf, is_training=True, interp_order=interp_order) # input pre-processing

    for case in range(train_conf['cases']):
        print("Start Case " + str(case) + " training...")
        print("Load data ...")
        # train_conf['this_case'] = case # this is used for saving sampled images
        input_data, labels = read_dataset(gen_conf, train_conf)
        print("Train model ...")
        model, mean, std, mse, corr = train_model(
            gen_conf, train_conf, input_data[:num_volumes[0]], labels[:num_volumes[0]], case)
        mean_array.append(mean)
        std_array.append(std)
        mse_array.append(mse)
        corr_array.append(corr)

    # write to file
    save_msecorr_array(gen_conf, train_conf, mse_array, corr_array)

    return mean_array, std_array, mse_array, corr_array

def train_model(gen_conf,
                train_conf,
                input_data,
                labels,
                case_name = 1):
    approach = train_conf['approach']
    dimension = train_conf['dimension']
    dataset_info = gen_conf['dataset_info'][train_conf['dataset']]
    num_classes = gen_conf['num_classes']
    num_modalities = dataset_info['modalities']
    output_shape = train_conf['output_shape'] ## output patch size (32, 32, 32)
    patch_shape = train_conf['patch_shape']  ## input patch size (32, 32, 8)

    if input_data.ndim == 6: # augmentation case
        num_samples = dataset_info['num_samples'][1]

    train_index, val_index = split_train_val(
        range(len(input_data)), train_conf['validation_split'])

    # mean, std = compute_statistics(input_data, num_modalities)
    # input_data = normalise_set(input_data, num_modalities, mean, std)

    if train_conf['num_epochs'] != 0:
        ## process input_data and labels here for dim = 6
        ## todo: could move this module to patch_utils with using a wrap function
        if input_data.ndim == 6: # the case of augmenting simulation
            x_train = np.zeros((0, num_modalities) + patch_shape)
            y_train = np.zeros((0, num_modalities) + output_shape)
            for smpl_idx in range(num_samples):
                this_input_data = np.reshape(input_data[:,:,0], input_data.shape[:2]+input_data.shape[3:])
                np.delete(input_data, 0, 2)
                this_x_train, this_y_train = overlap_patching(
                    gen_conf, train_conf, this_input_data[train_index], labels[train_index])
                x_train = np.vstack((x_train, this_x_train))
                y_train = np.vstack((y_train, this_y_train))
        else:
            x_train, y_train = overlap_patching(
                gen_conf, train_conf, input_data[train_index], labels[train_index])
        ##

        # debug_plot_patch(x_train, y_train)
        ## data standardization w.r.t. patch
        x_mean, x_std = compute_statistics(x_train, num_modalities)
        y_mean, y_std = compute_statistics(y_train, num_modalities)
        x_train = normalise_set(x_train, num_modalities, x_mean, x_std)
        y_train = normalise_set(y_train, num_modalities, y_mean, y_std)

        ## process input_data and labels here for dim = 6
        ## todo: could move this module to patch_utils with using a wrap function
        if input_data.ndim == 6:  # the case of augmenting simulation
            x_val = np.zeros((0, num_modalities) + patch_shape)
            y_val = np.zeros((0, num_modalities) + output_shape)
            for smpl_idx in range(num_samples):
                this_input_data = np.reshape(input_data[:,:,smpl_idx], input_data.shape[:2]+input_data.shape[3:])
                this_x_val, this_y_val = overlap_patching(
                gen_conf, train_conf, this_input_data[val_index], labels[val_index])
                x_val = np.vstack((x_val, this_x_val))
                y_val = np.vstack((y_val, this_y_val))
        else:
            x_val, y_val = overlap_patching(
                gen_conf, train_conf, input_data[val_index], labels[val_index])
            ##

        ## data standardization w.r.t. patch
        x_val = normalise_set(x_val, num_modalities, x_mean, x_std)
        y_val = normalise_set(y_val, num_modalities, y_mean, y_std)

        ## dictionary form of mean and std
        mean = {'input': x_mean, 'output': y_mean}
        std = {'input': x_std, 'output': y_std}
        ## save mean and std
        save_meanstd(gen_conf, train_conf, mean, std, case_name)

        # training callbacks
        callbacks = generate_callbacks(gen_conf, train_conf, case_name)
        if callbacks is not None: # train or not ?
            model = __train_model(
                gen_conf, train_conf, x_train, y_train, x_val, y_val, case_name, callbacks)
        else:
            model = read_model(gen_conf, train_conf, case_name)
            mean, std = read_meanstd(gen_conf, train_conf, case_name)

        ## compute mse and corr on validation set
        y_out = model.predict(x_val,
                              batch_size=train_conf['batch_size'],
                              verbose=train_conf['verbose'])
        mse = np.sum((y_out-y_val)**2)/y_val.shape[0]*y_std**2
        corr = 1-np.sum((y_out-y_val)**2)/np.sum((y_val)**2)
    else:
        model = []
        mean = {'input': 0, 'output': 0}
        std = {'input': 1, 'output': 1}
        mse = None
        corr = None

    return model, mean, std, mse, corr

def __train_model(gen_conf, train_conf, x_train, y_train, x_val, y_val, case_name, callbacks, vis_flag=False):
    model = generate_model(gen_conf, train_conf)

    print(model.summary()) # print model summary

    history = model.fit(
        x_train, y_train,
        batch_size=train_conf['batch_size'],
        epochs=train_conf['num_epochs'],
        validation_data=(x_val, y_val),
        verbose=train_conf['verbose'],
        callbacks=callbacks)
    print(history.params) # print model parameters

    # if vis_flag == True:
    visualize_model(model, history, gen_conf, train_conf, case_name, vis_flag)

    return model

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
