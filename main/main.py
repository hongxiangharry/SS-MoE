'''
    main.py: SR U-net for the IQT tutorial
    coder: H Lin
    Date: 30/06/19
    Version: v0.1.1
'''
# -------------------------- set gpu using tf ---------------------------
import sys
sys.path.append('..')

from config import general_configuration as gen_conf
from config import training_configuration as train_conf
from config import test_configuration as test_conf
from workflow.data_preparation import data_preparation
from workflow.build_training_set import build_training_set
from workflow.train import training
from workflow.test import testing
from workflow.evaluate import evaluating
import numpy as np

# data preparation
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

# # build training set
# count_patches = build_training_set(gen_conf, train_conf)

# training process
mse_array = training(gen_conf, train_conf)

# evaluation process
case_name = np.argmin(mse_array)
evaluating(gen_conf, train_conf, case_name=case_name)

# # test process
# case_name = np.argmin(mse_array)
# opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf)
# model = testing(gen_conf, test_conf, train_conf=None, case_name=case_name)
