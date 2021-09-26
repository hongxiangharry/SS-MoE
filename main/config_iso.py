general_configuration = {
    'num_classes' : 3, # label classes
    'dataset_path' : '', # in project folder
    'base_path' : '<PATH/TO/OUTPUT/RESULTS>', # in project folder
    'job_name' : 'default',
    'log_path_' : 'log',
    'model_path_' : 'models',
    'results_path_' : 'result',
    'evaluation_path_': 'evaluation',
    'dataset_info' : {
        'MUDI': {
            'format' : 'nii.gz',
            'dimensions': (82, 92, 56), # output shape
            'num_volumes': [5, 0], # train and test
            'real_modalities': 1344, # num volumes per 4D data
            'modalities': 1,
            'general_pattern': '{}/{}{}.{}',
            'patch_filename_pattern': '{}-{}-{}-{}-{}.zip',
            'outlier_detection_files':{
                'z-score': 'iso-interp_zscore.txt',
                'iqr': 'iso-interp_iqr.txt'
            },
            'path': ['<PROCESSED_SUPER_MUDI_DATA_DIR>',
                     '<SOURCE_SUPER_MUDI_DATA_DIR>',
                     '<TARGET_DIR_TO_UNZIP>/{}/patch',
                     '<SOURCE_ZIP_FILE_DIR>'],
            'in_postfix': 'iso', ## todo: look at this from dataset
            'out_postfix' : 'org',
            'modality_categories': ['MUDI'],
            'downsample_scale' : 2,
            'sparse_scale' : [2, 2, 2],
            'shrink_dim' : 3,
            'is_preproc': False, # input pre-processing
            'upsample_scale': [2, 2, 2],
            'interp_order' : 3 # try 0-5
        },
    }
}

training_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'IsoSRUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'MUDI',
    'dimension' : 3,
    'extraction_step' : (16, 16, 16),
    'extraction_step_test' :(8, 8, 8),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_training_patches' : 40000,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (16, 16, 16),
    'max_num_patches_per_file' : 10,
    'max_num_patches' : 20, # only for training
    'patch_sampling_rate' : 1,
    'bg_discard_percentage' : 0.5,
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
    'outlier_detection' : 'iqr',

}

test_configuration = {
    'retrain' : False,
    'activation' : 'null',
    'approach' : 'IsoSRUnet', # `SRUnet` or `AnisoUnet`
    'dataset' : 'MUDI',
    'dimension' : 3,
    'extraction_step' : (16, 16, 16),
    'extraction_step_test' :(8, 8, 8),
    'loss' : 'mean_squared_error',
    'metrics' : ['mse'],
    'batch_size' : 32,
    'num_epochs' : 100,
    'optimizer' : 'Adam',
    'output_shape' : (32, 32, 32),
    'output_shape_test' : (16, 16, 16),
    'patch_shape' : (16, 16, 16),
    'patch_sampling_rate': 1,
    'bg_discard_percentage' : 0.5,
    'patience' : 1000,
    'validation_split' : 0.20,
    'use_multiprocessing': False,
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
    'interp': {
        'is_interp': True,
        'interp_order': 3
    },
    'is_selfnormalised': False,
    'cases' : 1, # number of test cases
    'outlier_detection' : 'iqr',
}