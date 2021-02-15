general_configuration = {
    'num_classes' : 3, # label classes
    'dataset_path' : '', # PC
    'base_path' : '/cluster/project0/IQT_Nigeria/research/juntendo/results', # PC
    'job_name' : 'default', # 'srunet16_16_2_nf4' or 'anisounet16_16_2_nf4'
    'log_path_' : 'log',
    'model_path_' : 'models',
    'results_path_' : 'result',
    'evaluation_path_': 'evaluation',
    'dataset_info' : {
        'Juntendo-Volunteer': {
            'format' : 'nii',
            'dimensions': (192, 256, 256), # output shape
            'num_volumes': [3, 1], # train and test
            'modalities': 1,
            'general_pattern': '{}/{}{}',
            'path': '/cluster/project0/IQT_Nigeria/research/juntendo/data/process2',
            'in_postfixes': ['_reg.nii.gz'], # input postfix about modality
            'out_postfixes': ['.nii'], # output postfix about modality
            'subfolders' : ['LF', 'HF'], # [input sub-folder, output sub-folder]
            'modality_categories': ['T1w'],
            'downsample_scale' : 2,
            'sparse_scale' : [2, 1, 1],
            'shrink_dim' : 1,
            'is_preproc': False, # input pre-processing
            'upsample_scale': [2, 1, 1],
            'interp_order' : 3 # try 0-5
        },
    }
}

training_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'Juntendo-Volunteer',
    'dimension' : 3,
    'extraction_step' : (8, 16, 16),
    'extraction_step_test' :(8, 16, 16),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (16, 32, 32),
    'patch_sampling_rate' : 1, # only for training
    'bg_discard_percentage' : 0.2,
    'patience' : 5,
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
    'dataset' : 'Juntendo-Volunteer',
    'dimension' : 3,
    'extraction_step' : (8, 16, 16),
    'extraction_step_test' :(8, 16, 16),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (16, 32, 32),
    'bg_discard_percentage' : 0.2,
    'patience' : 5,
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
