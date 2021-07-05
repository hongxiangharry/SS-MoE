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
from config_iso_test import general_configuration as gen1_conf
from config_iso_test import test_configuration as test_conf
from workflow.data_preparation import data_preparation
from workflow.build_training_set import build_training_set
from workflow.train import training, training_S2, training_S3
from workflow.test_S2_train_label import generate_train_label
from workflow.test_S3_class import generate_class
from workflow.evaluate_S1_label import evaluating_S2_test_label
from workflow.evaluate import evaluating
import numpy as np
import pandas as pd
from workflow.test_S2 import test_S2_test
from workflow.test_S3 import test_S3_test



# test the in Stage3

opt, gen_conf_train, train_conf_S3_test = data_preparation(gen_conf, train_conf, 'train')
opt, gen_conf_test, test_conf_S3_test = data_preparation(gen_conf, test_conf, 'eval')
test_S3_test( gen_conf_train, gen_conf_test, train_conf_S3_test, test_conf_S3_test, flag='eval', case_name=0)

