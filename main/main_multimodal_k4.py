'''
    main.py: SR U-net for the IQT tutorial
    coder: H Lin
    Date: 30/06/19
    Version: v0.1.1
'''
# -------------------------- set gpu using tf ---------------------------

from config_multimodal_k4 import general_configuration as gen_conf
from config_multimodal_k4 import training_configuration as train_conf
from config_multimodal_k4 import test_configuration as test_conf
from workflow.data_preparation import data_preparation
from workflow.train import training
from workflow.test import testing
from workflow.evaluation import evaluation
from utils.preprocessing_util import preproc_dataset
import os
import numpy as np
from utils.ioutils import read_msecorr_array

# data preparation
# opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf)
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

# # default dataset
# dataset_name = train_conf['dataset']
# dataset_info = gen_conf['dataset_info'][dataset_name]
# dataset_info['training_subjects'] = [
# '200008', '200109', '200210', '200311', '200513', '200614', '200917', '201111',
# '201414', '201515', '201717', '201818', '202113', '202719', '202820']

# dataset_info['test_subjects'] = ['203418', '203721', '203923', '204016', '204218',
# '204319', '204420', '204521', '204622', '205119', '205220', '205725', '205826', '206222', '206323']

# GPU configuration on the Miller/Armstrong cluster
is_cmic_cluster = False

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

# # training process
# mean_array, std_array, mse_array, corr_array = training(gen_conf, train_conf)

mse_array, corr_array = read_msecorr_array(gen_conf, train_conf)

opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf)

case_name = np.argmax(corr_array)
model = testing(gen_conf, test_conf, train_conf=train_conf, case_name=case_name)
#
# evaluation
evaluation(gen_conf, test_conf, case_name=case_name)
