general_configuration = {
    'num_classes' : 3, # label classes
    'dataset_path' : '/cluster/project0/IQT_Nigeria/others/SuperMudi', # in project folder
    'base_path' : '/cluster/project0/IQT_Nigeria/others/SuperMudi/results', # in project folder
    'job_name' : 'default', # 'srunet16_16_2_nf4' or 'anisounet16_16_2_nf4'
    'log_path_' : 'log',
    'model_path_' : 'models',
    'results_path_' : 'result',
    'evaluation_path_': 'evaluation',
    'dataset_info' : {
        'MUDI': {
            'format' : 'nii.gz',
            'dimensions': (82, 92, 56), # output shape
            'num_volumes': [2, 3], # train and test
            'real_modalities': 1344, # num volumes per 4D data
            'modalities': 1,
            'general_pattern': '{}/{}{}.{}',
            'path': ['process', 'origin', 'patch'],
            'in_postfix': 'aniso', ## todo: look at this from dataset
            'out_postfix' : 'org',
            'modality_categories': ['MUDI'],
            'downsample_scale' : 2,
            'sparse_scale' : [1, 1, 2],
            'shrink_dim' : 3,
            'is_preproc': False, # input pre-processing
            'upsample_scale': [1, 1, 2],
            'interp_order' : 3 # try 0-5
        },
    }
}

training_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'MUDI',
    'dimension' : 3,
    'extraction_step' : (32, 32, 16),
    'extraction_step_test' :(16, 16, 8),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_training_patches' : 40000,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (32, 32, 16),
    'patch_sampling_rate' : 1, # only for training
    'bg_discard_percentage' : 0.2,
    'patience' : 1000,
    'validation_split' : 0.20,
    'use_multiprocessing' : False,
    'workers': 4,
    'verbose' : 1, # 0: save message flow in log, 1: process bar record, 2: epoch output record
    'shuffle' : True,
    'decay' : 0.000001,
    'learning_rate' : 0.001,
    'downsize_factor' : 1,
    'num_kernels' : 2,
    'num_filters' : 4,
    'mapping_times' : 2,
    'num_levels' : 3,
    'ishomo': False,
    'cases' : 1, # number of test cases
}

test_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'MUDI',
    'dimension' : 3,
    'extraction_step' : (32, 32, 16),
    'extraction_step_test' :(16, 16, 8),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (32, 32, 16),
    'bg_discard_percentage' : 0.2,
    'patience' : 1000,
    'validation_split' : 0.20,
    'verbose' : 1, # 0: save message flow in log, 1: process bar record, 2: epoch output record
    'shuffle' : True,
    'decay' : 0.000001,
    'learning_rate' : 0.001,
    'downsize_factor' : 1,
    'num_kernels' : 2,
    'num_filters' : 4,
    'mapping_times' : 2,
    'num_levels' : 3,
    'interp': {
        'is_interp': True,
        'interp_order': 3
    },
    'is_selfnormalised': False,
    'cases' : 1, # number of test cases
}