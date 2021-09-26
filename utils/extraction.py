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

from sklearn.feature_extraction.image import extract_patches as sk_extract_patches

def extract_patches(dimension, volume, patch_shape, extraction_step) :
    actual_patch_shape = patch_shape
    actual_extraction_step = extraction_step

    if dimension == 2 :
        if len(actual_patch_shape) == 3 :
            actual_patch_shape = actual_patch_shape[:1] + (1, ) + actual_patch_shape[1:]
            actual_extraction_step = actual_extraction_step[:1] + (1, ) + actual_extraction_step[1:]
        else :
            actual_patch_shape = (1, ) + actual_patch_shape
            actual_extraction_step = (1, ) + actual_extraction_step

    patches = sk_extract_patches(
        volume,
        patch_shape=actual_patch_shape,
        extraction_step=actual_extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)
