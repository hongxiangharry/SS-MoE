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
from utils.conf_utils import conf_dataset
from utils.conf_utils import set_conf_info
from utils.conf_utils import save_conf_info

def data_preparation(gen_conf, train_conf, trainTestFlag = 'train') :
    opt, gen_conf, train_conf = set_conf_info(gen_conf, train_conf)
    gen_conf, train_conf = conf_dataset(gen_conf, train_conf, trainTestFlag)
    save_conf_info(gen_conf, train_conf)

    return opt, gen_conf, train_conf