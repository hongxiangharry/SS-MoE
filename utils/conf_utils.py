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

    if dataset == 'MUDI':
        return conf_MUDI_dataset(gen_conf, train_conf, trainTestFlag)


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
