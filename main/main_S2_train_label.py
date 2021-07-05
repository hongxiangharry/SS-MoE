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
import numpy as np
import pandas as pd


# Generate the error label
# data preparation

opt, gen_conf_error_label, train_conf_error_label = data_preparation(gen_conf, train_conf)

case_name = 0
generate_train_label(gen_conf_error_label, train_conf_error_label)


save_csv_path = './csv_files'
raw_s2_label = pd.read_csv(save_csv_path + 'Stage2_label_train.csv')
index = raw_s2_label['index']
name_list = raw_s2_label['Name']
mse_list = raw_s2_label['MSE']
Level_1 = np.zeros((np.array(mse_list).shape))

quatier_1 = np.percentile(np.array(mse_list),25)
quatier_2 = np.percentile(np.array(mse_list),50)
quatier_3 = np.percentile(np.array(mse_list),75)

Level_1[mse_list>=quatier_1]=1
Level_1[mse_list>=quatier_2]=2
Level_1[mse_list>=quatier_3]=3

Data4stage2 = pd.DataFrame({'index':index, 'Name':name_list, 'MSE': mse_list, 'Level_1':Level_1})
Data4stage2.to_csv(save_csv_path + 'Stage2_label_train_1.csv', index = None, encoding='utf8')

