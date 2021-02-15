'''
    main.py: SR U-net for the IQT tutorial
    coder: H Lin
    Date: 30/06/19
    Version: v0.1.1
'''
# -------------------------- set gpu using tf ---------------------------
import sys
sys.path.append('..')

from config_flair_8x import general_configuration as gen_conf
from config_flair_8x import training_configuration as train_conf
from config_flair_8x import test_configuration as test_conf
from config_flair_8x import test_n19_configuration as test_n19_conf
from workflow.data_preparation import data_preparation
from workflow.interp import interpolate
from workflow.evaluation import evaluation, interp_eval

import os
import numpy as np

opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf)
interpolate(gen_conf, test_conf)
interp_eval(gen_conf, test_conf)

# n19 prediction
opt, gen_conf, test_n19_conf = data_preparation(gen_conf, test_n19_conf)
interpolate(gen_conf, test_n19_conf)
