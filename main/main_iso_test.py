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
from workflow.test_S2 import test_S2_test
from workflow.test_S3 import test_S3_test


'''
# pretraining
# data preparation
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)


# training process
mse_array = training(gen_conf, train_conf)

# test process

case_name = np.argmin(mse_array)

case_name = 0
opt, gen1_conf, test_conf = data_preparation(gen1_conf, test_conf, 'test')
testing(gen_conf, test_conf, case_name)
'''




# Generate the error label
# data preparation
'''
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

case_name = 0
#opt, gen1_conf, test_conf = data_preparation(gen1_conf, test_conf, 'test')
generate_train_label(gen_conf, train_conf)

# generate test error level label

opt, gen_conf, test_conf = data_preparation(gen_conf, test_conf, 'eval')
evaluating_S2_test_label(gen_conf, test_conf, flag='eval', case_name=0)
'''




'''
# Stage2-training fc
# data preparation
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

# training process
model, mse_array = training_S2(gen_conf, train_conf)
'''

# test the prediction in Stage2
'''
opt, gen_conf_train, train_conf = data_preparation(gen_conf, train_conf, 'train')
opt, gen_conf_test, test_conf = data_preparation(gen_conf, test_conf, 'eval')
test_S2_test( gen_conf_train, gen_conf_test, train_conf, test_conf, flag='eval', case_name=0)
'''



'''
# generate the classification prediction
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)
#opt, gen1_conf, test_conf = data_preparation(gen1_conf, test_conf, 'eval')
generate_class(gen_conf, train_conf)
'''







# Stage-3 training
opt, gen_conf, train_conf = data_preparation(gen_conf, train_conf)

# training process
mse_array = training_S3(gen_conf, train_conf)





'''
# test the in Stage3

opt, gen_conf_train, train_conf = data_preparation(gen_conf, train_conf, 'train')
opt, gen_conf_test, test_conf = data_preparation(gen_conf, test_conf, 'eval')
test_S3_test( gen_conf_train, gen_conf_test, train_conf, test_conf, flag='eval', case_name=0)
'''
