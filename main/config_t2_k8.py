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
        'HCP-Wu-Minn-Contrast-Multimodal': {
            'format' : 'nii',
            'dimensions': (260, 311, 256), # output shape
            'num_volumes': [15, 15], # train and test
            'modalities': 1,
            'general_pattern': '{}/{}_acpc_dc_restore_brain_sim036T_ds6_gap2{}.nii',
            'path': '/cluster/project0/IQT_Nigeria/HCPt1t2/sim',
            'in_postfix': ['_WM35.20_GM48.35'], ## todo: look at this from dataset
            'out_postfix' : '_groundtruth',
            'modality_categories': ['T2w', 'T1w', 'FLAIR', 'T2starw'],
            'downsample_scale' : 8,
            'sparse_scale' : [1, 1, 8],
            'shrink_dim' : 3,
            'is_preproc': False, # input pre-processing
            'upsample_scale': [1, 1, 8],
            'interp_order' : 3 # try 0-5
        },
        'Nigeria19-Multimodal': {
            'format' : 'nii',
            'dimensions': (309, 283, 148), # output shape
            'num_volumes': [0, 7], # train and test
            'modalities': 1,
            'general_pattern': '{}/{}{}_LF_{}_{}.nii',
            'path': '/cluster/project0/IQT_Nigeria/Nigeria_paired/process',
            'prefix': 'P',
            'in_postfix': 'SS',
            'out_postfix' : None,
            'modality_categories': ['T2', 'T1'],
            'downsample_scale' : 8,
            'sparse_scale' : [1, 1, 8],
            'shrink_dim' : 3,
            'is_preproc': False, # input pre-processing
            'upsample_scale': [1, 1, 8],
            'interp_order' : 3, # try 0-5
        },
    }
}

training_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'AnisoUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'HCP-Wu-Minn-Contrast-Multimodal',
    'dimension' : 3,
    'extraction_step' : (16, 16, 2),
    'extraction_step_test' :(16, 16, 2),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (32, 32, 4),
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
    'dataset' : 'HCP-Wu-Minn-Contrast-Multimodal',
    'dimension' : 3,
    'extraction_step' : (16, 16, 2),
    'extraction_step_test' :(16, 16, 2),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (32, 32, 8),
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
