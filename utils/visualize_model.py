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

from tensorflow.keras.utils import plot_model # keras utils
import matplotlib.pyplot as plt
from utils.callbacks import generate_callbacks, generate_output_filename
import os

def visualize_model(model, history, gen_conf, train_conf, case_name, vis_flag = False):
    plot_filename = generate_output_filename(
        gen_conf['model_path'],
        train_conf['dataset'],
        case_name,
        train_conf['approach'],
        train_conf['dimension'],
        str(train_conf['patch_shape']),
        str(train_conf['extraction_step']),
        'png')
    ## check and make folders
    plot_foldername = os.path.dirname(plot_filename)
    if not os.path.isdir(plot_foldername) :
        os.makedirs(plot_foldername)
    plot_model(model, to_file=plot_filename)

    if vis_flag == True:
        # Plot training & validation accuracy values
        plt.plot(history.history['mean_squared_error'])
        plt.plot(history.history['val_mean_squared_error'])
        plt.title('Model MSE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    return True