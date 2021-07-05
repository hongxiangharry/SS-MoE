'''
    main.py: SR U-net for the IQT tutorial
    coder: H Lin
    Date: 30/06/19
    Version: v0.1.1
'''
# -------------------------- set gpu using tf ---------------------------
import sys
sys.path.append('..')

from config_iso import general_configuration as gen_conf
from config_iso import training_configuration as train_conf
from config_iso import test_configuration as test_conf
from workflow.data_preparation import data_preparation
from workflow.build_training_set import build_training_set
from workflow.train import training
from workflow.test_S2 import test_S2_test
from workflow.evaluate_S1_label import evaluating_S2_test_label
import numpy as np


# generate test error level label

opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf, 'eval')
evaluating_S2_test_label(gen_conf, test_conf, flag='eval', case_name=0)
