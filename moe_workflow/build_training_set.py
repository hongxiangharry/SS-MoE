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
from utils.build_patch_lib_utils import build_patch_lib

## evaluate_using_training_testing_split
def build_training_set(gen_conf, train_conf) :
    count = build_patch_lib(gen_conf, train_conf)
    train_conf['actual_num_patches'] = count
    return count