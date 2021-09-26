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


# generate the classification prediction
opt, gen_conf_class, train_conf_class = data_preparation(gen_conf, train_conf)
#opt, gen1_conf, test_conf = data_preparation(gen1_conf, test_conf, 'eval')
generate_class(gen_conf_class, train_conf_class)

