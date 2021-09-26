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

import tensorflow.keras.backend as K
from tensorflow.keras.layers import ZeroPadding3D
import tensorflow as tf

class L2TV(object):
    def __init__(self, p=1.25, weight=1):
        # super().__init__(p, weight)
        super().__init__()
        self.p = p # power of tv
        self.weight = weight # regularisation parameter of tv term

    def l2_tv(self, y_true, y_pred):
        """Computes l2 loss + tv penalty.
        ```
        l2 = tf.keras.losses.MeanSquaredError()
        loss = l2(y_true, y_pred)
        ```
        where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
        Args:
          y_true: tensor of true targets.
          y_pred: tensor of predicted targets.
          delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        Returns:
          Tensor with
        """
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        tv = self.total_variation_loss(y_pred, self.p)
        loss = mse + self.weight * tv
        return loss

    def total_variation_loss(self, x, p=1.25):
        '''
        formula: https://raghakot.github.io/keras-vis/vis.regularizers/#totalvariation
        homogeneity: https://link.springer.com/content/pdf/10.1023/B:JMIV.0000011325.36760.1e.pdf
        :param x: 5D tensor, (batch, channel, x, y, z)
        :return:
        '''
        x = ZeroPadding3D(padding=1)(x)
        a = K.square( x[:, :, :-2, 1:-1, 1:-1] - x[:, :, 2:, 1:-1, 1:-1] )
        b = K.square( x[:, :, 1:-1, :-2, 1:-1] - x[:, :, 1:-1, 2:, 1:-1] )
        c = K.square( x[:, :, 1:-1, 1:-1, :-2] - x[:, :, 1:-1, 1:-1, 2:] )
        total = tf.math.reduce_sum(K.pow(a, p/2)) \
                + tf.math.reduce_sum(K.pow(b, p/2)) \
                + tf.math.reduce_sum(K.pow(c, p/2))
        return total

class L2L2(object):
    def __init__(self, weight=1):
        # super().__init__(p, weight)
        super().__init__()
        self.weight = weight # regularisation parameter of l2 term

    def l2_l2(self, y_true, y_pred):
        """Computes l2 loss + tv penalty.
        ```
        l2 = tf.keras.losses.MeanSquaredError()
        loss = l2(y_true, y_pred)
        ```
        where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
        Args:
          y_true: tensor of true targets.
          y_pred: tensor of predicted targets.
          delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        Returns:
          Tensor with
        """
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        l2 = self.l2_norm(y_pred)
        loss = mse + self.weight * l2
        return loss

    def l2_norm(self, x):
        '''
        formula: https://raghakot.github.io/keras-vis/vis.regularizers/#totalvariation
        homogeneity: https://link.springer.com/content/pdf/10.1023/B:JMIV.0000011325.36760.1e.pdf
        :param x: 5D tensor, (batch, channel, x, y, z)
        :return:
        '''
        total = tf.math.reduce_sum(K.square(x))
        return total