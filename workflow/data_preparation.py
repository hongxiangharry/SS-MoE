'''
    data_preparation.py:

'''

from utils.conf_utils import conf_dataset
from utils.conf_utils import set_conf_info
from utils.conf_utils import save_conf_info

def data_preparation(gen_conf, train_conf) :
    opt, gen_conf, train_conf = set_conf_info(gen_conf, train_conf)
    gen_conf, train_conf = conf_dataset(gen_conf, train_conf)
    save_conf_info(gen_conf, train_conf)

    return opt, gen_conf, train_conf