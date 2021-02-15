general_configuration = {
    'num_classes' : 3, # label classes
    'dataset_path' : '', # in project folder
    'base_path' : '/cluster/project0/IQT_Nigeria/research/baseline_t1t2', # in project folder
    'job_name' : 'default', # 'srunet16_16_2_nf4' or 'anisounet16_16_2_nf4'
    'log_path_' : 'log',
    'model_path_' : 'models',
    'results_path_' : 'result',
    'evaluation_path_': 'evaluation',
    'dataset_info' : {
        'MBB': {
            'format' : 'nii',
            'dimensions': (192, 512, 504), # output shape
            'num_volumes': [15, 15], # train and test
            'modalities': 1,
            'general_pattern': '{}_{}.nii',
            'path': '/cluster/project0/IQT_Nigeria/mbb-dataset/process',
            'subname_len': 6,
            'in_postfix': 'sim_norm', ## todo: look at this from dataset
            'out_postfix' : 'gt_norm',
            'modality_categories': ['FLAIR'],
            'downsample_scale' : 'multi',
            'sparse_scale' : [1, 2, 12],
            'shrink_dim' : 3,
            'is_preproc': False, # input pre-processing
            'upsample_scale': [1, 2, 12],
            'interp_order' : 3 # try 0-5
        },
        'Nigeria19-Multimodal': {
            'format' : 'nii.gz',
            'dimensions': (300, 300, 27), # output shape
            'num_volumes': [0, 12], # train and test
            'modalities': 1,
            'general_pattern': '{}/{}{}_LF_{}_{}.{}',
            'path': '/cluster/project0/IQT_Nigeria/Nigeria_paired/process',
            'prefix': 'P',
            'in_postfix': 'SS_resample',
            'out_postfix' : None,
            'modality_categories': ['FLAIR', 'T2', 'T1'],
            'downsample_scale' : 'multi',
            'sparse_scale' : [1, 2, 12],
            'shrink_dim' : 3,
            'is_preproc': False, # input pre-processing
            'upsample_scale': [1, 2, 12],
            'interp_order' : 3, # try 0-5
        },
    }
}

training_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'MBB',
    'dimension' : 3,
    'extraction_step' : (24, 12, 2),
    'extraction_step_test' :(24, 12, 2),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (48, 48, 48),
    'output_shape_test' : (24, 24, 24),
    'patch_shape' : (48, 24, 4),
    'patch_sampling_rate' : 1, # only for training
    'bg_discard_percentage' : 0.2,
    'patience' : 10,
    'validation_split' : 0.20,
    'verbose' : 1, # 0: save message flow in log, 1: process bar record, 2: epoch output record
    'shuffle' : True,
    'decay' : 0.000001,
    'learning_rate' : 0.001,
    'downsize_factor' : 1,
    'num_kernels' : 2,
    'num_filters' : 4,
    'mapping_times' : 2,
    'ishomo': False,
    'cases' : 1, # number of test cases
}

test_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'MBB',
    'dimension' : 3,
    'extraction_step' : (24, 12, 2),
    'extraction_step_test' :(24, 12, 2),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (48, 48, 48),
    'output_shape_test' : (24, 24, 24),
    'patch_shape' : (48, 24, 4),
    'bg_discard_percentage' : 0.2,
    'patience' : 10,
    'validation_split' : 0.20,
    'verbose' : 1, # 0: save message flow in log, 1: process bar record, 2: epoch output record
    'shuffle' : True,
    'decay' : 0.000001,
    'learning_rate' : 0.001,
    'downsize_factor' : 1,
    'num_kernels' : 2,
    'num_filters' : 4,
    'mapping_times' : 2,
    'ishomo': False,
    'is_selfnormalised': False,
    'cases' : 1, # number of test cases
}

test_n19_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'Nigeria19-Multimodal',
    'dimension' : 3,
    'extraction_step' : (24, 12, 2),
    'extraction_step_test' :(24, 12, 2),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (48, 48, 48),
    'output_shape_test' : (24, 24, 24),
    'patch_shape' : (48, 24, 4),
    'bg_discard_percentage' : 0.2,
    'patience' : 10,
    'validation_split' : 0.20,
    'verbose' : 1, # 0: save message flow in log, 1: process bar record, 2: epoch output record
    'shuffle' : True,
    'decay' : 0.000001,
    'learning_rate' : 0.001,
    'downsize_factor' : 1,
    'num_kernels' : 2,
    'num_filters' : 4,
    'mapping_times' : 2,
    'ishomo': False,
    'is_selfnormalised': False,
    'cases' : 1, # number of test cases
}