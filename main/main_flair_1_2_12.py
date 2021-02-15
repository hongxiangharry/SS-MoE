'''
    main.py: SR U-net for the IQT tutorial
    coder: H Lin
    Date: 30/06/19
    Version: v0.1.1
'''
# -------------------------- set gpu using tf ---------------------------
import sys
sys.path.append('..')

from config_flair_1_2_12 import general_configuration as gen_conf
from config_flair_1_2_12 import training_configuration as train_conf
from config_flair_1_2_12 import test_configuration as test_conf
from config_flair_1_2_12 import test_n19_configuration as test_n19_conf
from workflow.data_preparation import data_preparation
from workflow.train import training
from workflow.test import testing
from workflow.evaluation import evaluation
from utils.preprocessing_util import preproc_dataset
import os
import numpy as np

# data preparation
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

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

# training process
mean_array, std_array, mse_array, corr_array = training(gen_conf, train_conf)

case_name = np.argmin(mse_array)
opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf)
model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)

# evaluation
evaluation(gen_conf, test_conf, case_name=case_name)

# n19 prediction
case_name = np.argmin(mse_array)
opt, gen_conf, test_n19_conf = data_preparation(gen_conf, test_n19_conf)
model = testing(gen_conf, test_n19_conf, train_conf=train_conf, case_name=case_name)