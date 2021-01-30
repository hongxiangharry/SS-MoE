import os
import argparse
import csv
import numpy as np
from numpy.random import randint
from IPython.core.debugger import set_trace
import glob

def argument_parse():
    # -------------------------- Set up configurations ----------------------------
    # Basic settings
    ## description: text to display before the argument help
    parser = argparse.ArgumentParser(description='IQT-Keras-version')
    ## dest : The name of the attribute to be added to the object returned by parse_args()
    ## If there is no explicit written "dest" parameter, the key should be "e" in this case.

    # default: None for '?' and [] for '*'
    # list to tuple
    # system conf
    parser.add_argument('--gpu', type=str, default="0", help='which GPU to use')

    ## directory
    parser.add_argument('-dp', '--dataset_path', dest='dataset_path', nargs='?', type=str, help='dataset directory')
    parser.add_argument('-bp', '--base_path', dest='base_path', nargs='?', type=str, help='workplace directory')
    parser.add_argument('-jn', '--job_name', dest='job_name', nargs='?', type=str, help='job name of folder')
    parser.add_argument('-j', '--job_id', dest='job_id', nargs='?', default=None, type=str, help='job id to qsub system')

    ## dataset info
    parser.add_argument('--dataset', dest='dataset', nargs='?', type=str, help='dataset name')
    parser.add_argument('--no_subject', dest='no_subject', nargs='*', type=int, help='set train/test subjects')
    parser.add_argument('--num_samples', dest='num_samples', nargs='*', type=int, help='set augmenting total/train/test samples')

    # patching info
    parser.add_argument('-es', '--extraction_step', dest='extraction_step', nargs='*', type=int,
                        help='stride between patch for training')
    parser.add_argument('-est', '--extraction_step_test', dest='extraction_step_test', nargs='*', type=int,
                        help='stride between patch for testing')
    parser.add_argument('-ip', '--input_patch', dest='input_patch', nargs='*', type=int,
                        help='input patch shape')
    parser.add_argument('-op', '--output_patch', dest='output_patch', nargs='*', type=int,
                        help='output patch shape')
    parser.add_argument('-opt', '--output_patch_test', dest='output_patch_test', nargs='*', type=int,
                        help='output patch shape for testing')
    parser.add_argument('-mnp', '--max_num_patches', dest='max_num_patches', nargs='?', type=int,
                        help='maximal number of patches per volume')
    parser.add_argument('-ntp', '--num_training_patches', dest='num_training_patches', nargs='?', type=int,
                        help='number of training patches')
    parser.add_argument('-od', '--outlier_detection', dest='outlier_detection', nargs='?', type=str, help='Use an outlier detection method')
    parser.add_argument('--rebuild', action='store_true', help='rebuild training patch set')

    # network info
    parser.add_argument('--approach', dest='approach', nargs='?', type=str, help='name of network architecture')
    parser.add_argument('-l', '--loss', dest='loss', nargs='?', type=str, help='name of loss')
    parser.add_argument('-lp', '--loss_params', dest='loss_params', nargs='*', type=float, help='parameters for loss function')
    parser.add_argument('-ne', '--no_epochs', dest='no_epochs', nargs='?', type=int, help='number of epochs')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', nargs='?', type=int, help='batch size')
    parser.add_argument('--patience', dest='patience', nargs=1, type=int, help='early stop at patience number')
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', nargs='?', type=float, help='learning rate')
    parser.add_argument('-dc', '--decay', dest='decay', nargs='?', type=float, help='decay of learning rate')
    parser.add_argument('-dsf', '--downsize_factor', dest='downsize_factor', nargs='?', type=int, help='downsize factor for CNN')
    parser.add_argument('-nk', '--num_kernels', dest='num_kernels', nargs='?', type=int, help='number of kernels per block')
    parser.add_argument('-nf', '--num_filters', dest='num_filters', nargs='?', type=int,
                        help='number of filters per conv layer')
    parser.add_argument('-mt', '--mapping_times', dest='mapping_times', nargs='?', type=int,
                        help='number of FSRCNN shrinking layers')
    parser.add_argument('-nl', '--num_levels', dest='num_levels', nargs='?', type=int,
                        help='number of U-Net levels')
    parser.add_argument('-cas', '--ca_scale', dest='ca_scale', nargs='?', type=float,
                        help='shrinkage scale of channel attention module')
    parser.add_argument('-carr', '--ca_reduced_rate', dest='ca_reduced_rate', nargs='?', type=int,
                        help='reduced rate of filters in channel attention module')

    parser.add_argument('-c', '--cases', dest='cases', nargs='?', type=int, help='number of training cases')
    parser.add_argument('-vs', '--validation_split', dest='validation_split', nargs='?', type=float, help='validation split rate')


    # action : Turn on the value for the key, i.e. "overwrite=True"
    parser.add_argument('--retrain', action='store_true', help='restart the training completely')

    # parser.add_argument('--continue', action='store_true', help='continue training from previous epoch')
    # parser.add_argument('--is_reset', action='store_true', help='reset the patch library?')
    # parser.add_argument('--not_save', action='store_true', help='invoke if you do not want to save the output')
    # parser.add_argument('--disp', action='store_true', help='save the displayed outputs?')
    #
    # # Directories:
    # parser.add_argument('--base_dir', type=str, default='/home/harrylin/experiments', help='base directory')
    # parser.add_argument('--gt_dir', type=str, default='/SAN/vision/hcp/DCA_HCP.2013.3_Proc',
    #                     help='ground truth directory')
    # parser.add_argument('--subpath', type=str, default='T1w/Diffusion', help='subdirectory in gt_dir')
    # parser.add_argument('--mask_dir', type=str, default='/SAN/vision/hcp/Ryu/miccai2017/hcp_masks',
    #                     help='directory of segmentation masks')
    # parser.add_argument('--mask_subpath', type=str, default='', help='subdirectory in mask_dir')
    #u
    # # Network
    # parser.add_argument('-m', '--method', dest='method', type=str, default='espcn', help='network type')
    # parser.add_argument('--no_filters', type=int, default=50, help='number of initial filters')
    # parser.add_argument('--no_layers', type=int, default=2, help='number of hidden layers')
    # parser.add_argument('--is_shuffle', action='store_true',
    #                     help='Needed for ESPCN/DCESPCN. Want to reverse shuffle the HR output into LR space?')
    # parser.add_argument('--is_BN', action='store_true', help='want to use batch normalisation?')
    # parser.add_argument('--optimizer', type=str, default='adam', help='optimization method')
    # parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default='1e-3',
    #                     help='learning rate')
    # parser.add_argument('-dr', '--dropout_rate', dest='dropout_rate', type=float, default='0.0', help='drop-out rate')
    # parser.add_argument('--no_epochs', type=int, default=200, help='number of epochs to train for')
    # parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    # parser.add_argument('--validation_fraction', type=float, default=0.5, help='fraction of validation data')
    # parser.add_argument('--valid', action='store_true', help='pick the best model based on the loss, not the MSE?')
    #
    # # Data/task
    # parser.add_argument('--is_map', action='store_true', help='Want to use MAP-MRI instead?')
    # parser.add_argument('-pl', '--no_patches', dest='no_patches', type=int, default=2250,
    #                     help='number of patches sampled from each train subject')
    # parser.add_argument('-ts', '--no_subjects', dest="no_subjects", type=int, default='8', help='background value')
    # parser.add_argument('--no_channels', type=int, default=6, help='number of channels')
    # parser.add_argument('-bgval', '--background_value', dest="background_value", type=float, default='0',
    #                     help='background value')
    # parser.add_argument('-us', '--upsampling_rate', dest="upsampling_rate", type=int, default=2, help='upsampling rate')
    # parser.add_argument('-ir', '--input_radius', dest="input_radius", type=int, default=5, help='input radius')
    # parser.add_argument('--pad_size', type=int, default=-1,
    #                     help='size of padding applied before patch extraction. Set -1 to apply maximal padding.')
    # parser.add_argument('--is_clip', action='store_true',
    #                     help='want to clip the images (0.1% - 99.9% percentile) before patch extraction? ')
    # parser.add_argument('--patch_sampling_opt', type=str, default='default',
    #                     help='sampling scheme for patch extraction')
    # parser.add_argument('--transform_opt', type=str, default='standard', help='normalisation transform')
    # parser.add_argument('-pp', '--postprocess', dest='postprocess', action='store_true',
    #                     help='post-process the estimated highres output?')

    arg = parser.parse_args()
    return vars(arg)  ## return a dictionary type of arguments and the values.

def set_conf_info(gen_conf, train_conf):
    opt = argument_parse() # read parser from the command line

    if opt['dataset_path'] is not None: gen_conf['dataset_path'] = opt['dataset_path']
    if opt['base_path'] is not None: gen_conf['base_path'] = opt['base_path']
    if opt['job_name'] is not None: gen_conf['job_name'] = opt['job_name']
    if opt['job_id'] is not None: gen_conf['job_id'] = opt['job_id']

    if opt['dataset'] is not None: train_conf['dataset'] = opt['dataset']
    if opt['no_subject'] is not None: gen_conf['dataset_info'][train_conf['dataset']]['num_volumes'] = opt['no_subject']
    if opt['num_samples'] is not None: gen_conf['dataset_info'][train_conf['dataset']]['num_samples'] = opt['num_samples']

    if opt['extraction_step'] is not None: train_conf['extraction_step'] = tuple(opt['extraction_step'])
    if opt['extraction_step_test'] is not None: train_conf['extraction_step_test'] = tuple(opt['extraction_step_test'])
    if opt['input_patch'] is not None: train_conf['patch_shape'] = tuple(opt['input_patch'])
    if opt['output_patch'] is not None: train_conf['output_shape'] = tuple(opt['output_patch'])
    if opt['output_patch_test'] is not None: train_conf['output_shape_test'] = tuple(opt['output_patch_test'])
    if opt['max_num_patches'] is not None: train_conf['max_num_patches'] = opt['max_num_patches']
    if opt['num_training_patches'] is not None: train_conf['num_training_patches'] = opt['num_training_patches']
    train_conf['outlier_detection'] = opt['outlier_detection'] if opt['outlier_detection'] is not None else None
    train_conf['rebuild'] = opt['rebuild']

    if opt['approach'] is not None: train_conf['approach'] = opt['approach']
    if opt['loss'] is not None: train_conf['loss'] = opt['loss']
    if opt['loss_params'] is not None: train_conf['loss_params'] = opt['loss_params']
    if opt['no_epochs'] is not None: train_conf['num_epochs'] = opt['no_epochs']
    if opt['batch_size'] is not None: train_conf['batch_size'] = opt['batch_size']
    if opt['patience'] is not None: train_conf['patience'] = opt['patience']

    if opt['learning_rate'] is not None: train_conf['learning_rate'] = opt['learning_rate']
    if opt['decay'] is not None: train_conf['decay'] = opt['decay']
    if opt['downsize_factor'] is not None: train_conf['downsize_factor'] = opt['downsize_factor']
    if opt['num_kernels'] is not None: train_conf['num_kernels'] = opt['num_kernels']
    if opt['num_filters'] is not None: train_conf['num_filters'] = opt['num_filters']
    if opt['mapping_times'] is not None: train_conf['mapping_times'] = opt['mapping_times']
    if opt['num_levels'] is not None: train_conf['num_levels'] = opt['num_levels']
    if opt['ca_scale'] is not None: train_conf['ca_scale'] = opt['ca_scale']
    if opt['ca_reduced_rate'] is not None: train_conf['ca_reduced_rate'] = opt['ca_reduced_rate']
    if opt['validation_split'] is not None: train_conf['validation_split'] = opt['validation_split']

    if opt['cases'] is not None: train_conf['cases'] = opt['cases']

    train_conf['retrain'] = opt['retrain']
    return opt, gen_conf, train_conf


def conf_dataset(gen_conf, train_conf, trainTestFlag = 'train'):
    # configure log/model/result/evaluation paths.
    gen_conf['log_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['log_path_'])
    gen_conf['model_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['model_path_'])
    gen_conf['results_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['results_path_'])
    gen_conf['evaluation_path'] = os.path.join(gen_conf['base_path'], gen_conf['job_name'], gen_conf['evaluation_path_'])

    dataset = train_conf['dataset']

    if dataset == 'HBN':
        return conf_HBN_dataset(gen_conf, train_conf)
    if dataset == 'HCP-Wu-Minn-Contrast' :
        return conf_HCPWuMinnContrast_dataset(gen_conf, train_conf)
    if dataset == 'HCP-Wu-Minn-Contrast-Augmentation' :
        return conf_HCPWuMinnContrastAugmentation_dataset(gen_conf, train_conf)
    if dataset in ['HCP-Wu-Minn-Contrast-Multimodal', 'Nigeria19-Multimodal']:
        return conf_Multimodal_dataset(gen_conf, train_conf)
    if dataset == 'Juntendo-Volunteer' :
        return conf_Juntendo_Volunteer_dataset(gen_conf, train_conf)
    if dataset == 'MBB':
        return conf_MBB_dataset(gen_conf, train_conf)
    if dataset == 'MUDI':
        return conf_MUDI_dataset(gen_conf, train_conf, trainTestFlag)

'''
def conf_MUDI_dataset(gen_conf, traintest_conf, trainTestFlag = 'train'):
    dataset_path = gen_conf['dataset_path']
    dataset_name = traintest_conf['dataset']
    job_id = gen_conf['job_id'] if 'job_id' in gen_conf.keys() else None

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path'][0]

    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    in_postfix = dataset_info['in_postfix']

    whole_dataset_path = os.path.join(dataset_path, path)
    if trainTestFlag == 'train' or trainTestFlag == 'eval':
        subject_lib = sorted([os.path.basename(subject) for subject in glob.glob(whole_dataset_path+'/cdmri*')])
    elif trainTestFlag == 'test':
        subject_lib = sorted([os.path.basename(os.path.dirname(subject)) for subject in glob.glob(whole_dataset_path + '/testsbj*/' + in_postfix + '.zip')])
            # sorted([os.path.basename(os.path.dirname(subject)) for subject in glob.glob('/cluster/project0/IQT_Nigeria/others/SuperMudi/process/testsbj*/aniso.zip')])

    # subject_lib = sorted(os.listdir(hcp_dataset_path)) # alphabetical order
    # # for cluster
    # if '.DS_Store' in subject_lib:
    #     subject_lib.remove('.DS_Store')
    print(np.array(subject_lib).shape)
    subject_lib_specfic = subject_lib[4]
    print(len(subject_lib_specfic))
    print(train_num_volumes)
    print(test_num_volumes)
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    if trainTestFlag == 'train' :
        dataset_info['training_subjects'] = []
        for idx, subject in enumerate(subject_lib_specfic):
            if os.path.isdir(os.path.join(whole_dataset_path, subject)) \
                    and idx < dataset_info['num_volumes'][0]:
                dataset_info['training_subjects'].append(subject)

    if trainTestFlag == 'test' or trainTestFlag == 'eval':
        dataset_info['test_subjects'] = []
        idx = dataset_info['num_volumes'][0]
        for subject in subject_lib_specfic[dataset_info['num_volumes'][0]:]:
            if os.path.isdir(os.path.join(whole_dataset_path, subject)) \
                    and idx < dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]:
                dataset_info['test_subjects'].append(subject)
                idx += 1

    # set patch lib temp path to local storage .. [25/08/20]
    if job_id is not None:
        dataset_info['path'][2] = dataset_info['path'][2].format(job_id)

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, traintest_conf
'''



def conf_MUDI_dataset(gen_conf, traintest_conf, trainTestFlag = 'train'):
    dataset_path = gen_conf['dataset_path']
    dataset_name = traintest_conf['dataset']
    job_id = gen_conf['job_id'] if 'job_id' in gen_conf.keys() else None

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path'][0]

    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    in_postfix = dataset_info['in_postfix']

    whole_dataset_path = os.path.join(dataset_path, path)
    if trainTestFlag == 'train' or trainTestFlag == 'eval':
        subject_lib = sorted([os.path.basename(subject) for subject in glob.glob(whole_dataset_path+'/cdmri*')])
    elif trainTestFlag == 'test':
        subject_lib = sorted([os.path.basename(os.path.dirname(subject)) for subject in glob.glob(whole_dataset_path + '/testsbj*/' + in_postfix + '.zip')])
            # sorted([os.path.basename(os.path.dirname(subject)) for subject in glob.glob('/cluster/project0/IQT_Nigeria/others/SuperMudi/process/testsbj*/aniso.zip')])

    # subject_lib = sorted(os.listdir(hcp_dataset_path)) # alphabetical order
    # # for cluster
    # if '.DS_Store' in subject_lib:
    #     subject_lib.remove('.DS_Store')
    print(np.array(subject_lib).shape)
    #subject_lib = subject_lib[4]
    print(len(subject_lib))
    print(train_num_volumes)
    print(test_num_volumes)
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    if trainTestFlag == 'train' :
        dataset_info['training_subjects'] = []
        for idx, subject in enumerate(subject_lib):
            if os.path.isdir(os.path.join(whole_dataset_path, subject)) \
                    and idx < dataset_info['num_volumes'][0]:
                dataset_info['training_subjects'].append(subject)

    if trainTestFlag == 'test' or trainTestFlag == 'eval':
        dataset_info['test_subjects'] = []
        idx = dataset_info['num_volumes'][0]
        for subject in subject_lib[dataset_info['num_volumes'][0]:]:
            if os.path.isdir(os.path.join(whole_dataset_path, subject)) \
                    and idx < dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]:
                dataset_info['test_subjects'].append(subject)
                idx += 1

    # set patch lib temp path to local storage .. [25/08/20]
    if job_id is not None:
        dataset_info['path'][2] = dataset_info['path'][2].format(job_id)

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, traintest_conf



def conf_MBB_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]
    in_postfix = dataset_info['in_postfix']
    subname_len = dataset_info['subname_len']

    whole_dataset_path = os.path.join(dataset_path, path, pattern).format('*', in_postfix)
    subject_lib = sorted([os.path.splitext(os.path.basename(filepath))[0][:subname_len] for filepath in glob.glob(whole_dataset_path)])
    print(subject_lib,subname_len)

    # subject_lib = sorted(os.listdir(hcp_dataset_path)) # alphabetical order
    # # for cluster
    # if '.DS_Store' in subject_lib:
    #     subject_lib.remove('.DS_Store')
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if idx_sn < dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf

def conf_Juntendo_Volunteer_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]
    subfolder_hf = dataset_info['subfolders'][1] # high-field folder

    whole_dataset_path = os.path.join(dataset_path, path, subfolder_hf)
    subject_lib = sorted([os.path.splitext(os.path.basename(subject))[0] for subject in glob.glob(whole_dataset_path+'/*')])

    # subject_lib = sorted(os.listdir(hcp_dataset_path)) # alphabetical order
    # # for cluster
    # if '.DS_Store' in subject_lib:
    #     subject_lib.remove('.DS_Store')
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if idx_sn < dataset_info['num_volumes'][0] + dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf

def conf_Multimodal_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    whole_dataset_path = os.path.join(dataset_path, path)
    subject_lib = sorted([os.path.basename(subject) for subject in glob.glob(whole_dataset_path+'/*')])

    # subject_lib = sorted(os.listdir(hcp_dataset_path)) # alphabetical order
    # # for cluster
    # if '.DS_Store' in subject_lib:
    #     subject_lib.remove('.DS_Store')
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if os.path.isdir(os.path.join(whole_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if os.path.isdir(os.path.join(whole_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0] + \
                dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf

def conf_HCPWuMinnContrastAugmentation_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    modalities = dataset_info['modalities']
    modality_categories = dataset_info['modality_categories']  # T1w, T2w, FLAIR, T2starw
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]
    out_postfix = dataset_info['out_postfix']  ## '_sim036T_ds6_gap2_groundtruth'
    train_samples = dataset_info['num_samples'][0] ## total number of simulation samples for training
    test_samples = dataset_info['num_samples'][2] ## total number of simulation samples for test ## 1

    hcp_dataset_path = os.path.join(dataset_path, path)
    subject_lib = sorted(os.listdir(hcp_dataset_path)) # alphabetical order
    # for cluster
    if '.DS_Store' in subject_lib:
        subject_lib.remove('.DS_Store')
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    dataset_info['training_in_postfix'] = np.zeros((train_num_volumes, modalities, train_samples), dtype=object) #  three dimensional list
    idx_sn = 0
    for sub_idx, subject in enumerate(subject_lib):
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            print(subject)
            for mod_idx in range(modalities):
                # random sampling per subject
                tmp_filepath = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject,
                                                            modality_categories[mod_idx],
                                                            '')
                commonprefix = os.path.splitext(os.path.basename(tmp_filepath))[0]
                ## common prefix ## T1w_acpc_dc_restore_brain

                len_commonprefix = len(commonprefix)  ## length of common prefix
                curr_dir = os.path.dirname(tmp_filepath)  ## current path, '/home/harrylin/HCP_aug/200109/T1w'
                allfiles = [os.path.splitext(f)[0] for f in os.listdir(curr_dir) if
                            os.path.isfile(os.path.join(curr_dir, f))]  ## search all files under the current directory
                ## debug:
                # print('All files under the current folder.')
                # print(allfiles) # 'T1w_acpc_dc_restore_brain_sim036T_ds6_gap2_WM63_GM50'
                in_postfixes = [f[len_commonprefix:] for f in allfiles if
                                f[:len_commonprefix] == commonprefix and f[len_commonprefix:] != out_postfix]
                # dataset_info['training_in_postfix'][sub_idx][mod_idx] = in_postfixes[randint(len(in_postfixes))]
                for idx in range(train_samples):
                    dataset_info['training_in_postfix'][sub_idx][mod_idx][idx] = in_postfixes[idx]
            idx_sn += 1

    dataset_info['test_subjects'] = []
    dataset_info['test_in_postfix'] =  np.zeros((test_num_volumes, modalities, test_samples), dtype=object) # three dimensional numpy list
    for sub_idx, subject in enumerate(subject_lib[idx_sn:]):
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0] + \
                dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            print(subject)
            # random sampling per subject
            for mod_idx in range(modalities):
                # random sampling per subject
                tmp_filepath = os.path.join(dataset_path,
                                            path,
                                            pattern).format(subject,
                                                            modality_categories[mod_idx],
                                                            '')
                commonprefix = os.path.splitext(os.path.basename(tmp_filepath))[0]
                ## common prefix ## T1w_acpc_dc_restore_brain

                len_commonprefix = len(commonprefix)  ## length of common prefix
                curr_dir = os.path.dirname(tmp_filepath)  ## current path, '/home/harrylin/HCP_aug/200109/T1w'
                allfiles = [os.path.splitext(f)[0] for f in os.listdir(curr_dir) if
                            os.path.isfile(os.path.join(curr_dir, f))]  ## search all files under the current directory
                ## debug:
                # print('All files under the current folder.')
                # print(allfiles) # 'T1w_acpc_dc_restore_brain_sim036T_ds6_gap2_WM63_GM50'
                in_postfixes = [f[len_commonprefix:] for f in allfiles if
                                f[:len_commonprefix] == commonprefix and f[len_commonprefix:] != out_postfix]
                # dataset_info['test_in_postfix'][sub_idx][mod_idx] = in_postfixes[randint(len(in_postfixes))]
                for idx in range(test_samples):
                    dataset_info['test_in_postfix'][sub_idx][mod_idx][idx] = in_postfixes[idx]
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf


def conf_HBN_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    hcp_dataset_path = os.path.join(dataset_path, path)
    subject_lib = os.listdir(hcp_dataset_path)
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0] + \
                dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf

def conf_HCPWuMinnContrast_dataset(gen_conf, train_conf):
    dataset_path = gen_conf['dataset_path']
    dataset_name = train_conf['dataset']

    dataset_info = gen_conf['dataset_info'][dataset_name]
    path = dataset_info['path']
    train_num_volumes = dataset_info['num_volumes'][0]
    test_num_volumes = dataset_info['num_volumes'][1]

    hcp_dataset_path = os.path.join(dataset_path, path)
    subject_lib = sorted(os.listdir(hcp_dataset_path)) # alphabetical order
    # for cluster
    if '.DS_Store' in subject_lib:
        subject_lib.remove('.DS_Store')
    assert len(subject_lib) >=  train_num_volumes + test_num_volumes

    dataset_info['training_subjects'] = []
    idx_sn = 0
    for subject in subject_lib:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0]:
            dataset_info['training_subjects'].append(subject)
            idx_sn += 1

    dataset_info['test_subjects'] = []
    for subject in subject_lib[idx_sn:]:
        if os.path.isdir(os.path.join(hcp_dataset_path, subject)) \
                and idx_sn < dataset_info['num_volumes'][0] + \
                dataset_info['num_volumes'][1]:
            dataset_info['test_subjects'].append(subject)
            idx_sn += 1

    gen_conf['dataset_info'][dataset_name] = dataset_info

    return gen_conf, train_conf

def save_conf_info(gen_conf, train_conf):
    dataset_name = train_conf['dataset']
    dataset_info = gen_conf['dataset_info'][dataset_name]

    # check and create parent folder
    csv_filename_gen = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        '_gen_conf',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'cvs')
    csv_filename_train = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        '_train_conf', # todo: should be different if random sampling
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'cvs')
    csv_filename_dataset = generate_output_filename(
        gen_conf['log_path'],
        train_conf['dataset'],
        '_dataset',
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'cvs')
    ## check and make folders
    csv_foldername = os.path.dirname(csv_filename_gen)
    if not os.path.isdir(csv_foldername) :
        os.makedirs(csv_foldername)

    # save gen_conf
    with open(csv_filename_gen, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, gen_conf.keys())
        w.writeheader()
        w.writerow(gen_conf)

    with open(csv_filename_train, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, train_conf.keys())
        w.writeheader()
        w.writerow(train_conf)

    with open(csv_filename_dataset, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, dataset_info.keys())
        w.writeheader()
        w.writerow(dataset_info)

def generate_output_filename(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension):
    file_pattern = '{}/{}/{}-{}-{}-{}-{}.{}'
    return file_pattern.format(path, dataset, case_name, approach, dimension, patch_shape, extraction_step, extension)
