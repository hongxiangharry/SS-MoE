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

import numpy as np

def pad_both_sides(dimension, vol, pad) :
    pad_func = lambda vol, pad : np.pad(vol, pad, 'constant', constant_values=0)

    if dimension == 2 :
        pad = (0, ) + pad

    padding = ((pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2]))

    if len(vol.shape) == 3 :
        return pad_func(vol, padding)
    else :
        return pad_func(vol, ((0, 0),) + padding)