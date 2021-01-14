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
from workflow.data_preparation import data_preparation
from workflow.build_training_set import build_training_set

# data preparation
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

# build training set
count_patches = build_training_set(gen_conf, train_conf)
