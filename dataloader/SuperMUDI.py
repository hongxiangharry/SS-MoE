from tensorflow.keras.utils import Sequence
import numpy as np
import os
import zipfile
import glob
import math
import shutil

def DefineTrainValSuperMudiDataloader(gen_conf, train_conf, is_shuffle_trainval = False):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    validation_split_ratio = train_conf['validation_split']

    path = dataset_info['path'][2]  # patch
    patch_dir = os.path.join(dataset_path, path)

    N_vol = len(sorted(glob.glob(patch_dir + '/*.npz')))
    if is_shuffle_trainval is True:
        shuffle_order = np.random.permutation(N_vol)
    else:
        shuffle_order = np.arange(N_vol)

    N = train_conf['num_training_patches']  # num of used training patches

    val_volumes = np.int32(np.ceil(N_vol * validation_split_ratio))
    train_volumes = N_vol - val_volumes

    N_val = np.int32(np.ceil(N * validation_split_ratio))
    N_train = N - N_val

    trainDataloader = SuperMudiSequence(gen_conf, train_conf, shuffle_order[:train_volumes], N_train)
    if validation_split_ratio != 0:
        valDataloader = SuperMudiSequence(gen_conf, train_conf, shuffle_order[train_volumes:], N_val)
    else:
        valDataloader = None

    return trainDataloader, valDataloader

class SuperMudiSequence(Sequence):
    def __init__(self, gen_conf, train_conf, shuffle_order, N):
        dataset = train_conf['dataset']
        dataset_path = gen_conf['dataset_path']
        dataset_info = gen_conf['dataset_info'][dataset]

        patch_shape = train_conf['patch_shape']
        extraction_step = train_conf['extraction_step']

        self.batch_size = train_conf['batch_size']
        self.N = N
        validation_split = train_conf['validation_split']

        # It is an update for local storage setup. All required data have been completely unzipped to the local storage. Need to set up dataset_info->dataset->path[2]
        path = dataset_info['path'][2]  # patch
        self.patch_dir = os.path.join(dataset_path, path)

        self.is_shuffle = train_conf['shuffle']

        ## old version
        # self.patch_zip_path = os.path.join(dataset_path, path, "{}-{}.zip").format(patch_shape, extraction_step)
        # self.patch_dir = os.path.splitext(self.patch_zip_path)[0]

        # # unzip packages
        # if not os.path.isdir(self.patch_dir):
        #     self.__unzip(self.patch_zip_path, self.patch_dir)
        ## old version

        # define shuffle list outside
        self.patch_lib_filenames = np.array(sorted(glob.glob(self.patch_dir + '/*.npz')))
        self.patch_lib_filenames = self.patch_lib_filenames[shuffle_order]

        # # train-val split
        # self.patch_lib_filenames, self.N = self.__split_train_val(self.patch_lib_filenames, self.N, validation_split, is_val_gen)
        # expanded patch lib = N
        self.patch_lib_filenames = np.tile(self.patch_lib_filenames, math.ceil(self.N / len(self.patch_lib_filenames)))[:self.N]

        # random shuffle
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)

    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __getitem__(self, idx):
        batch_filenames = self.patch_lib_filenames[idx * self.batch_size:(idx+1)*self.batch_size]

        x_batch = []
        y_batch = []
        for filename in batch_filenames:
            x_patches, y_patches = self.__read_patch_data(filename)
            rnd_patch_idx = np.random.randint(len(x_patches), size=1)
            x_batch.append(x_patches[np.asscalar(rnd_patch_idx)])
            y_batch.append(y_patches[np.asscalar(rnd_patch_idx)])
        # print(np.array(x_batch).shape, np.array(x_batch).dtype)
        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        # random shuffle
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)

    def __unzip(self, src_filename, dest_dir):
        with zipfile.ZipFile(src_filename) as z:
            z.extractall(path=dest_dir)

    def clear_extracted_files(self):
        if os.path.isdir(self.patch_dir):
            shutil.rmtree(self.patch_dir)
            return True
        else:
            print("'{}' doesn't exist ...".format(self.patch_dir))
            return False

    def __read_patch_data(self, filepath):
        files = np.load(filepath)
        return files['x_patches'], files['y_patches']

    def __split_train_val(self, train_indexes, N, validation_split, is_val_gen = False):
        N_vol = len(train_indexes)
        val_volumes = np.int32(np.ceil(N_vol * validation_split))
        train_volumes = N_vol - val_volumes

        val_patches = np.int32(np.ceil(N * validation_split))
        train_patches = N-val_patches

        if is_val_gen is True and validation_split != 0:
            return train_indexes[train_volumes:], val_patches
        elif is_val_gen is False:
            return train_indexes[:train_volumes], train_patches # training data
        else:
            raise ValueError("validation_split should be non-zeroed value when is_val_gen == True")

    # def __kfold_split_train_val

