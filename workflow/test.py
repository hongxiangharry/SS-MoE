import numpy as np
from architectures.arch_creator import generate_model
from utils.callbacks import generate_callbacks
from utils.ioutils import read_dataset, save_volume, read_model, generate_output_filename, read_meanstd, read_meanstd_MUDI_output
from utils.reconstruction import reconstruct_volume_imaging, reconstruct_volume_imaging4, reconstruct_volume_imaging3
from utils.patching_utils import overlap_patching
from utils.preprocessing_util import preproc_input

def testing(gen_conf, test_conf, case_name = 0) :
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]

    ## interpolation
    if dataset_info['is_preproc'] == True:
        interp_order = dataset_info['interp_order']
        print("Interpolating test data...")
        preproc_input(gen_conf, test_conf, is_training=False, interp_order=interp_order) # input pre-processing

    print("Start testing ... load data and model...")
    ## load data
    input_data, _ = read_dataset(gen_conf, test_conf, 'test')

    # mean, std = read_meanstd(gen_conf, conf, case_name) # for patch
    # mean_out, std_out = read_meanstd_MUDI_output(gen_conf, conf) # for output 4d image
    model = read_model(gen_conf, test_conf, case_name)

    print("Test model ...")
    ## test and save output
    # test_model(gen_conf, test_conf, input_data, model, None, None, mean_out, std_out, case_name)
    test_model(gen_conf, test_conf, input_data, model, None, None, None, None, case_name)

    return model

# latest one
def test_model(gen_conf,
               test_conf,
               input_data,
               model,
               mean,
               std,
               mean_out,
               std_out,
               case_name = 0) :
    num_classes = gen_conf['num_classes']
    output_shape = test_conf['output_shape']
    dataset_info = gen_conf['dataset_info'][test_conf['dataset']]
    dimensions = dataset_info['dimensions']
    num_volumes = dataset_info['num_volumes'][1]
    num_modalities = dataset_info['modalities'] # designed number of modalities, a particular setup in test
    real_num_modalities = dataset_info['real_modalities']
    batch_size = test_conf['batch_size']

    for vol_idx in range(0, num_volumes) :
        recon_im_4d = np.zeros(dimensions+(real_num_modalities, ))
        for mod_idx in range(real_num_modalities):
            test_index = vol_idx * real_num_modalities + mod_idx # index of data
            ## should only sample one subject
            if input_data.ndim == 6 :
                input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:2]+input_data.shape[3:])
            else:
                input_data_temp = np.reshape(input_data[test_index], (1, )+input_data.shape[1:])

            x_test, _ = overlap_patching(gen_conf, test_conf, input_data_temp,
                         output_data = None,
                         trainTestFlag = 'test',
                         representative_modality = 0)

            # x_test = normalise_volume(x_test, num_modalities, mean['input'], std['input'])

            recon_im = model.predict(x_test, verbose=test_conf['verbose'])
            # de-normalisation before prediction
            # recon_im = denormalise_volume(recon_im, num_modalities, mean['output'], std['output'])

            print("Reconstructing ...")
            recon_im_4d[:, :, :, mod_idx] = reconstruct_volume_imaging4(gen_conf, test_conf, recon_im[:, 0])
            del x_test, recon_im

        # image-level
        # recon_im_4d = denormalise_4d_volume(recon_im_4d, mean_out, std_out) # cancel denormalise, update [31/08/20]
        save_volume(gen_conf, test_conf, recon_im_4d, (vol_idx, case_name) )

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
