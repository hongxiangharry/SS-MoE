'''
    main.py: SR U-net for the IQT tutorial
    coder: H Lin
    Date: 30/06/19
    Version: v0.1.1
'''
# -------------------------- set gpu using tf ---------------------------

from config import general_configuration as gen_conf
from config import training_configuration as train_conf
from config import test_configuration as test_conf
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
# # default dataset
# dataset_name = train_conf['dataset']
# dataset_info = gen_conf['dataset_info'][dataset_name]
# dataset_info['training_subjects'] = [
# '200008', '200109', '200210', '200311', '200513', '200614', '200917', '201111',
# '201414', '201515', '201717', '201818', '202113', '202719', '202820']

# dataset_info['test_subjects'] = ['203418', '203721', '203923', '204016', '204218',
# '204319', '204420', '204521', '204622', '205119', '205220', '205725', '205826', '206222', '206323']

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
gen_conf['job_name'] = '140719_augHCP_Aniso_ns1'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 0
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 2
test_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_augHCP_Aniso_ns2'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 4
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 4
test_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_augHCP_Aniso_ns4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 5
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 8
test_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_augHCP_Aniso_ns8'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 4
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# SRUnet
test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_augHCP_SRUnet_ns1'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
## save dataset info
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
##
# training sample: 1
case_name = 9
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 2
test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_augHCP_SRUnet_ns2'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 1
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 4
test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_augHCP_SRUnet_ns4'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 3
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# training sample: 8
test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_augHCP_SRUnet_ns8'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 0
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)




# SRUnet

# training sample: 1
from config_fixed import general_configuration as gen_conf
from config_fixed import training_configuration as train_conf
from config_fixed import test_configuration as test_conf


# AnisoUnet

train_conf['approach'] = 'AnisoUnet'
test_conf['approach'] = 'AnisoUnet'
gen_conf['job_name'] = '140719_HCP_Aniso'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
opt, gen_conf, train_conf  = data_preparation(gen_conf, train_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 0
model = testing(gen_conf, test_conf, train_conf=train_conf, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

# SRUnet

train_conf['approach'] = 'SRUnet'
test_conf['approach'] = 'SRUnet'
gen_conf['job_name'] = '140719_HCP_SRUnet'
opt, gen_conf, test_conf  = data_preparation(gen_conf, test_conf)
opt, gen_conf, train_conf  = data_preparation(gen_conf, train_conf)
gen_conf['dataset_info']['HCP-Wu-Minn-Contrast-Augmentation'] = dataset_info
case_name = 0
model = testing(gen_conf, test_conf, train_conf=train_conf, case_name=case_name)
evaluation(gen_conf, test_conf, case_name=case_name)

