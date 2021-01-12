import numpy as np

from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks
from utils.ioutils import read_dataset, save_volume, save_volume_MICCAI2012, read_model, generate_output_filename, read_meanstd
from utils.reconstruction import reconstruct_volume_imaging, reconstruct_volume_imaging4, reconstruct_volume_imaging3
from utils.patching_utils import overlap_patching
from utils.preprocessing_util import preproc_input

def testing(gen_conf, test_conf, train_conf = None, case_name = 0) :
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]

    num_volumes = dataset_info['num_volumes']
    num_modalities = dataset_info['modalities']
    dimension = test_conf['dimension']
    patch_shape = test_conf['patch_shape']
    output_shape = test_conf['output_shape']
    is_selfnormalised = test_conf['is_selfnormalised']

    ## interpolation
    if dataset_info['is_preproc'] == True:
        interp_order = dataset_info['interp_order']
        print("Interpolating test data...")
        preproc_input(gen_conf, test_conf, is_training=False, interp_order=interp_order) # input pre-processing

    print("Start testing ... load data, mean, std and model...")
    ## load data
    input_data, _ = read_dataset(gen_conf, test_conf, 'test')

    # ## case_name // num_modality
    # case_name = case_name // num_modalities

    if train_conf is None:
        conf = test_conf
    else:
        conf = train_conf

    mean, std = read_meanstd(gen_conf, conf, case_name)
    model = read_model(gen_conf, conf, case_name)

    if is_selfnormalised == True:
        print("Self normalize data ...")
        x_mean, x_std = compute_statistics(input_data, num_modalities)
        input_data = normalise_volume(input_data, num_modalities, x_mean, x_std)

    print("Test model ...")
    ## test and save output
    test_model_3(gen_conf, test_conf, input_data, model, mean, std, case_name)  # comment cauz' debugging
    # test_model(gen_conf, test_conf, input_data, model) # comment cauz' debugging
    # test_model_2(gen_conf, test_conf, input_data) # debugging

    return model

# latest one
def test_model_3(gen_conf,
                 test_conf,
                 input_data,
                 model,
                 mean,
                 std,
                 case_name = 0) :
    num_classes = gen_conf['num_classes']
    output_shape = test_conf['output_shape']
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]
    num_volumes = dataset_info['num_volumes'][1]
    num_modalities = dataset_info['modalities']
    batch_size = test_conf['batch_size']
    is_selfnormalised = test_conf['is_selfnormalised']

    test_indexes = range(0, num_volumes)
    for idx, test_index in enumerate(test_indexes) :
        # mean, std = compute_statistics(input_data[test_index], num_modalities)
        # input_data[test_index] = normalise_volume(input_data[test_index], num_modalities, mean['train'], std['train'])

        # if patch_shape != output_shape :
        #     pad_size = ()
        #     for dim in range(dimension) :
        #         pad_size += (output_shape[dim], )
        #     test_vol = pad_both_sides(dimension, test_vol, pad_size)
        #
        # x_test = build_testing_set(gen_conf, test_conf, test_vol)

        ## should only sample one subject
        if input_data.ndim == 6 :
            input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:2]+input_data.shape[3:])
        else:
            input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:])

        x_test, _ = overlap_patching(gen_conf, test_conf, input_data_temp,
                     output_data = None,
                     trainTestFlag = 'test',
                     representative_modality = 0)

        # normalisation before prediction
        if is_selfnormalised == False:
            x_test = normalise_volume(x_test, num_modalities, mean['input'], std['input'])

        recon_im = model.predict(x_test,
                                 batch_size=batch_size,
                                 verbose=test_conf['verbose'])
        # de-normalisation before prediction
        recon_im = denormalise_volume(recon_im, num_modalities, mean['output'], std['output'])

        # recon_im = recon_im.reshape((len(recon_im),) + output_shape + (num_classes,))
        for idx2 in range(num_modalities) :
            print("Reconstructing ...")
            # new feature
            # recon_im2 = reconstruct_volume_imaging3(gen_conf, test_conf, recon_im[:, idx2])
            recon_im2 = reconstruct_volume_imaging4(gen_conf, test_conf, recon_im[:, idx2])
            save_volume(gen_conf, test_conf, recon_im2, (idx, idx2, case_name))
        del x_test, recon_im

    return True

def test_model_2(gen_conf,
                 test_conf,
                 input_data,
                 model = None) :
    num_classes = gen_conf['num_classes']
    output_shape = test_conf['output_shape']
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]
    num_volumes = dataset_info['num_volumes'][1]
    num_modalities = dataset_info['modalities']
    batch_size = test_conf['batch_size']

    test_indexes = range(0, num_volumes)
    for idx, test_index in enumerate(test_indexes) :
        # mean, std = compute_statistics(input_data[test_index], num_modalities)
        # input_data[test_index] = normalise_volume(input_data[test_index], num_modalities, mean, std)

        # if patch_shape != output_shape :
        #     pad_size = ()
        #     for dim in range(dimension) :
        #         pad_size += (output_shape[dim], )
        #     test_vol = pad_both_sides(dimension, test_vol, pad_size)
        #
        # x_test = build_testing_set(gen_conf, test_conf, test_vol)
        input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:])
        x_test, _ = overlap_patching(gen_conf, test_conf, input_data_temp,
                     output_data = None,
                     trainTestFlag = 'test',
                     representative_modality = 0)
        recon_im = x_test
        # recon_im = recon_im.reshape((len(recon_im),) + output_shape + (num_classes,))
        for idx2 in range(num_modalities) :
            print("Reconstructing ...")
            recon_im2 = reconstruct_volume_imaging(gen_conf, test_conf, recon_im[:, idx2])
            save_volume(gen_conf, test_conf, recon_im2, (idx, idx2))

        del x_test, recon_im

    return True

def test_model(gen_conf,
               test_conf,
               input_data,
               model) :
    num_classes = gen_conf['num_classes']
    output_shape = test_conf['output_shape']
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]
    num_volumes = dataset_info['num_volumes'][1]
    num_modalities = dataset_info['modalities']
    batch_size = test_conf['batch_size']

    test_indexes = range(0, num_volumes)
    for idx, test_index in enumerate(test_indexes) :
        # mean, std = compute_statistics(input_data[test_index], num_modalities)
        # input_data[test_index] = normalise_volume(input_data[test_index], num_modalities, mean, std)

        # if patch_shape != output_shape :
        #     pad_size = ()
        #     for dim in range(dimension) :
        #         pad_size += (output_shape[dim], )
        #     test_vol = pad_both_sides(dimension, test_vol, pad_size)
        #
        # x_test = build_testing_set(gen_conf, test_conf, test_vol)
        input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:])
        x_test, _ = overlap_patching(gen_conf, test_conf, input_data_temp,
                     output_data = None,
                     trainTestFlag = 'test',
                     representative_modality = 0)
        recon_im = model.predict(x_test,
                                 batch_size=batch_size,
                                 verbose=test_conf['verbose'])
        # recon_im = recon_im.reshape((len(recon_im),) + output_shape + (num_classes,))
        for idx2 in range(num_modalities) :
            print("Reconstructing ...")
            recon_im2 = reconstruct_volume_imaging(gen_conf, test_conf, recon_im[:, idx2])
            save_volume(gen_conf, test_conf, recon_im2, (idx, idx2))

        del x_test, recon_im

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

def compute_statistics(input_data, num_modalities) :
    mean = np.zeros((num_modalities, ))
    std = np.zeros((num_modalities, ))

    for modality in range(num_modalities) :
        modality_data = input_data[:, modality]
        mean[modality] = np.mean(modality_data)
        std[modality] = np.std(modality_data)

    return mean, std
