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

from utils.preprocessing_util import interp_input

def interpolate(gen_conf, test_conf) :
    if 'interp' in test_conf:
        interp = test_conf['interp']
        is_interp = interp['is_interp']
    else:
        is_interp = False

    ## interpolation
    if is_interp is True:
        interp_order = interp['interp_order']
        print("Interpolating test data...")
        interp_input(gen_conf, test_conf, interp_order=interp_order) # input pre-processing

    return True