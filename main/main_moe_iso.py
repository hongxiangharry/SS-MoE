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

from config_moe_iso import general_configuration as gen_conf
from config_moe_iso import training_configuration as train_conf
from config_moe_iso import test_configuration as test_conf
from moe_workflow.data_preparation import data_preparation
from moe_workflow.build_training_set import build_training_set
from moe_workflow.train import train_moe
from moe_workflow.test import testing
from moe_workflow.evaluate import evaluating
import numpy as np

# data preparation
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

# # build training set
# count_patches = build_training_set(gen_conf, train_conf)

# training process
mse_array = train_moe(gen_conf, train_conf)

# test process
case_name = np.argmin(mse_array)
opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf, 'eval')
evaluating(gen_conf, test_conf, flag='eval', case_name=case_name)
