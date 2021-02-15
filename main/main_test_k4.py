'''
    main.py: SR U-net for the IQT tutorial
    coder: H Lin
    Date: 30/06/19
    Version: v0.1.1
'''
# -------------------------- set gpu using tf ---------------------------

from config_k4 import general_configuration as gen_conf
from config_k4 import training_configuration as train_conf
from config_k4 import test_configuration as test_conf
from workflow.data_preparation import data_preparation
from workflow.train import training
from workflow.test import testing
from workflow.evaluation import evaluation
from utils.preprocessing_util import preproc_dataset
import os

# data preparation
# opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
dataset_info = gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation']

# GPU configuration on the Miller/Armstrong cluster
is_cmic_cluster = True

if is_cmic_cluster == True:
    # GPUs devices:
    ## Marco "CUDA_VISIBLE_DEVICES" defines the working GPU in the CMIC clusters.
    gpu_no = opt['gpu']
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())  ## Check the GPU list.

    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    KTF.set_session(session)


# k = 8, c = 5-10, training sample = 1/2/4/8
# multiple sample, 2 networks

# AnisoUnet

# training sample: 1
test_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_augHCP_Aniso_ns1_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 0
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 2
test_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_augHCP_Aniso_ns2_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 0
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 4
test_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_augHCP_Aniso_ns4_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 1
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 8
test_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_augHCP_Aniso_ns8_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 1
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# SRUnet

test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_augHCP_SRUnet_ns1_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
# training sample: 1
case_name = 1
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 2
test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_augHCP_SRUnet_ns2_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 1
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 4
test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_augHCP_SRUnet_ns4_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 1
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 8
test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_augHCP_SRUnet_ns8_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 3
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# SRUnet

# training sample: 1
from config_fixed_k4 import general_configuration as gen_conf
from config_fixed_k4 import training_configuration as train_conf
from config_fixed_k4 import test_configuration as test_conf

# AnisoUnet
test_conf['approach'] = 'AnisoUnet'
train_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_HCP_Aniso_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
opt, gen_conf, train_conf  = data_preparation(gen_conf, train_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 0
model = testing(gen_conf, test_conf, train_conf=train_conf, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

test_conf['approach'] = 'SRUnet'
train_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_HCP_SRUnet_k4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
opt, gen_conf, train_conf  = data_preparation(gen_conf, train_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 2
model = testing(gen_conf, test_conf, train_conf=train_conf, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)


