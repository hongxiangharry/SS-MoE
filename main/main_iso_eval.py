'''

    Project: Self-supervised Mixture of Experts
    Publication: "Generalised Super Resolution for Quantitative MRI Using Self-supervised Mixture of Experts" published in MICCAI 2021.
    Authors: Hongxiang Lin, Yukun Zhou, Paddy J. Slator, Daniel C. Alexander
    Affiliation: Centre for Medical Image Computing, Department of Computer Science, University College London
    Email to the corresponding author: [Hongxiang Lin] harry.lin@ucl.ac.uk
    Date: 26/09/21
    Version: v1.0.1
    License: MIT

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

'''
# generate test error level label

opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf, 'eval')
evaluating_S2_test_label(gen_conf, test_conf, flag='eval', case_name=0)
'''


# test in stage2
opt, gen_conf_train, train_conf = data_preparation(gen_conf, train_conf)
opt, gen_conf_test, test_conf = data_preparation(gen_conf, test_conf, 'eval')
test_S2_test(gen_conf_train, gen_conf_test, train_conf, test_conf, flag='eval', case_name=0)

