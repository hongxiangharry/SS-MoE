from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import os
import zipfile
import glob
import math
import shutil
import pandas as pd


'''
def DefineTrainValSuperMudiDataloader(gen_conf, train_conf, is_shuffle_trainval = False):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    validation_split_ratio = train_conf['validation_split']

    path = dataset_info['path'][2]  # patch
    patch_dir = os.path.join(dataset_path, path)

    # get the path of the patch and the N_vol is 5
    N_vol = len(sorted(glob.glob(patch_dir + '/*.npz')))
    if is_shuffle_trainval is True:
        shuffle_order = np.random.permutation(N_vol)
    else:
        shuffle_order = np.arange(N_vol)

    # the num_training_patch is 40000, says there are 40000 patch seg out for training
    N = train_conf['num_training_patches']  # num of used training patches

    # the ratio 0.2 vol (1) for validation, and 0.8 vol (4) for training
    val_volumes = np.int32(np.ceil(N_vol * validation_split_ratio))
    train_volumes = N_vol - val_volumes

    # the ratio 0.2 patch (8000) for validation, and 0.8 patch (32000) for training
    N_val = np.int32(np.ceil(N * validation_split_ratio))
    N_train = N - N_val

    # shuffle_order is arrange, so no shuffle here, 32000 patch 
    trainDataloader = SuperMudiSequence(gen_conf, train_conf, shuffle_order[:train_volumes], N_train)
    if validation_split_ratio != 0:
        
        # shuffle_order is False, so 32000-4000 patch for val
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

        # batch[0] name is /scratch0/harrylin/2700564/patch/240-cdmri0014-(8, 8, 8)-(4, 4, 4)-0008.npz
        #print('batch name is,', batch_filenames[0])

        x_batch = []
        y_batch = []
        for filename in batch_filenames:

            # filename is /scratch0/harrylin/2700564/patch/632-cdmri0011-(8, 8, 8)-(4, 4, 4)-0009.npz
            #print('filename is: ', filename)
            x_patches, y_patches = self.__read_patch_data(filename)

            # patch length is 10
            #print('patch length is: ', len(x_patches))
            rnd_patch_idx = np.random.randint(len(x_patches), size=1)

            # pick 1 out of the 10
            #print('idx,', rnd_patch_idx)
            x_batch.append(x_patches[np.asscalar(rnd_patch_idx)])
            y_batch.append(y_patches[np.asscalar(rnd_patch_idx)])
            
            # (37, 1, 8, 8, 8)
            # (37, 1, 16, 16, 16)
            #print(np.shape(x_batch))
            #print(np.shape(y_batch))
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


'''



def DefineTrainValSuperMudiDataloader(gen_conf, train_conf, para, csv_path, is_shuffle_trainval = False):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    validation_split_ratio = train_conf['validation_split']

    path = dataset_info['path'][2]  # patch
    patch_dir = os.path.join(dataset_path, path)

    data_list = pd.read_csv(csv_path)
    #label_in = data_list[['Name']]
    patch_lib_filenames_in = np.array(data_list[['Name']])
    N_vol = len(patch_lib_filenames_in)

    # get the path of the patch and the N_vol is 5
    #N_vol = len(sorted(glob.glob(patch_dir + '/*.npz')))
    #N_vol = train_conf['num_training_patches']
    if is_shuffle_trainval is True:
        shuffle_order = np.random.permutation(N_vol)
    else:
        shuffle_order = np.arange(N_vol)

    # the num_training_patch is 40000, says there are 40000 patch seg out for training
    #N = train_conf['num_training_patches']  # num of used training patches
    N = len(patch_lib_filenames_in)
    # the ratio 0.2 vol (1) for validation, and 0.8 vol (4) for training
    val_volumes = np.int32(np.ceil(N_vol * validation_split_ratio))
    train_volumes = N_vol - val_volumes

    # the ratio 0.2 patch (8000) for validation, and 0.8 patch (32000) for training
    N_val = np.int32(np.ceil(N * validation_split_ratio))
    N_train = N - N_val

    # shuffle_order is arrange, so no shuffle here, 32000 patch 
    trainDataloader = SuperMudiSequence(gen_conf, train_conf, para, shuffle_order[:train_volumes], N_train, csv_path)
    if validation_split_ratio != 0:
        
        # shuffle_order is False, so 32000-4000 patch for val
        valDataloader = SuperMudiSequence(gen_conf, train_conf, para, shuffle_order[train_volumes:], N_val, csv_path)
    else:
        valDataloader = None

    return trainDataloader, valDataloader



class SuperMudiSequence(Sequence):
    def __init__(self, gen_conf, train_conf, para, shuffle_order, N, csv_path):
        dataset = train_conf['dataset']
        dataset_path = gen_conf['dataset_path']
        dataset_info = gen_conf['dataset_info'][dataset]

        patch_shape = train_conf['patch_shape']
        extraction_step = train_conf['extraction_step']

        self.para = para
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
        '''
        self.patch_lib_filenames = np.array(sorted(glob.glob(self.patch_dir + '/*.npz')))
        self.patch_lib_filenames = self.patch_lib_filenames[shuffle_order]

        # # train-val split
        # self.patch_lib_filenames, self.N = self.__split_train_val(self.patch_lib_filenames, self.N, validation_split, is_val_gen)
        # expanded patch lib = N
        self.patch_lib_filenames = np.tile(self.patch_lib_filenames, math.ceil(self.N / len(self.patch_lib_filenames)))[:self.N]

        # random shuffle
        # 20210124
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)
        '''

        data_list = pd.read_csv(csv_path)
        
        patch_lib_filenames_in = np.array(data_list['Name'])

        #print('!!!!!!!!!!!!!!!!!')
        #print(patch_lib_filenames_in[2:10])
        #print(label_level_list_in[2:10])

        self.patch_lib_filenames = patch_lib_filenames_in[shuffle_order]

        #print('@@@@@@@@@@@@@@@@@')
        #print(self.patch_lib_filenames[2:10])
        #print(self.label_level_list[2:10])
        
        para_file=open('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/parameters.txt')
        txt=para_file.readlines()
        para_list=[]
        for w in txt:
            #w=w.replace('\n','')
            w=w.split()
            w = list(map(float, w))
            para_list.append(w)

        self.para_numpy = np.array(para_list)

        d_mean = np.mean(self.para_numpy, axis=0)
        d_std = np.std(self.para_numpy,axis=0)

        self.para_numpy = (self.para_numpy-d_mean)/d_std

        self.n_classes = 4

    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __getitem__(self, idx):

        batch_filenames = self.patch_lib_filenames[idx * self.batch_size:(idx+1)*self.batch_size]

        #print('###########################')
        #print(batch_filenames)
        #print(batch_label)


        # batch[0] name is /scratch0/harrylin/2700564/patch/240-cdmri0014-(8, 8, 8)-(4, 4, 4)-0008.npz
        #print('batch name is,', batch_filenames[0])

        batch_error_label = []
        x_batch = []
        y_batch = []
        para_batch = []
        
        for filename in batch_filenames:
            #rnd_patch_idx += 1
            # filename is /scratch0/harrylin/2700564/patch/632-cdmri0011-(8, 8, 8)-(4, 4, 4)-0009.npz
            #print('filename is: ', filename)

            filename_detail = filename.split('-patch-')[0] + '.npz'
            x_patches, y_patches = self.__read_patch_data(filename_detail)

            specific_patch_num = filename.split('-patch-')[1]
            x_batch.append(x_patches[int(specific_patch_num)])
            y_batch.append(y_patches[int(specific_patch_num)])


        #print(np.array(batch_filenames))
        #print(np.array(batch_label))

        if self.para=='para':
            return [np.array(x_batch), np.squeeze(np.array(para_batch))], np.array(y_batch)
        else:
            return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        # random shuffle
        #self.patch_lib_filenames = self.patch_lib_filenames
        #20210124
        #if self.is_shuffle is True:
        #    np.random.shuffle(self.patch_lib_filenames)
        self.patch_lib_filenames = self.patch_lib_filenames

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





def DefineTrainValSuperMudiDataloader_new(gen_conf, train_conf, is_shuffle_trainval = False):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    validation_split_ratio = train_conf['validation_split']

    path = dataset_info['path'][2]  # patch
    patch_dir = os.path.join(dataset_path, path)

    # get the path of the patch and the N_vol is 5
    N_vol = len(sorted(glob.glob(patch_dir + '/*.npz')))
    if is_shuffle_trainval is True:
        shuffle_order = np.random.permutation(N_vol)
    else:
        shuffle_order = np.arange(N_vol)

    # the num_training_patch is 40000, says there are 40000 patch seg out for training
    N = train_conf['num_training_patches']  # num of used training patches

    # the ratio 0.2 vol (1) for validation, and 0.8 vol (4) for training
    val_volumes = np.int32(np.ceil(N_vol * validation_split_ratio))
    train_volumes = N_vol - val_volumes

    # the ratio 0.2 patch (8000) for validation, and 0.8 patch (32000) for training
    N_val = np.int32(np.ceil(N * validation_split_ratio))
    N_train = N - N_val

    # shuffle_order is arrange, so no shuffle here, 32000 patch 
    trainDataloader = SuperMudiSequence_new(gen_conf, train_conf, shuffle_order[:train_volumes], N_train)
    if validation_split_ratio != 0:
        
        # shuffle_order is False, so 32000-4000 patch for val
        valDataloader = SuperMudiSequence_new(gen_conf, train_conf, shuffle_order[train_volumes:], N_val)
    else:
        valDataloader = None

    return trainDataloader, valDataloader



class SuperMudiSequence_new(Sequence):
    def __init__(self, gen_conf, train_conf, shuffle_order, N):
        dataset = train_conf['dataset']
        dataset_path = gen_conf['dataset_path']
        dataset_info = gen_conf['dataset_info'][dataset]

        self.patch_idx = 0

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
        # 20210124
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)

    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __getitem__(self, idx):
        batch_filenames = self.patch_lib_filenames[idx * self.batch_size:(idx+1)*self.batch_size]

        # batch[0] name is /scratch0/harrylin/2700564/patch/240-cdmri0014-(8, 8, 8)-(4, 4, 4)-0008.npz
        #print('batch name is,', batch_filenames[0])

        x_batch = []
        y_batch = []
        name_list = []
        patch_idx_list = []

        #patch_idx = 0

        for filename in batch_filenames:
            self.patch_idx += 1
            # filename is /scratch0/harrylin/2700564/patch/632-cdmri0011-(8, 8, 8)-(4, 4, 4)-0009.npz
            #print('filename is: ', filename)
            x_patches, y_patches = self.__read_patch_data(filename)

            # patch length is 10
            #print('patch length is: ', len(x_patches))
            rnd_patch_idx = np.random.randint(len(x_patches), size=1)
            
            filename_detail = filename.split('.np')[0] + '-patch-{}'.format(np.asscalar(rnd_patch_idx)) 
            #file_dir = file_dir.split('.')[0]+'.png'
            # pick 1 out of the 10
            #print('idx,', rnd_patch_idx)
            x_batch.append(x_patches[np.asscalar(rnd_patch_idx)])
            y_batch.append(y_patches[np.asscalar(rnd_patch_idx)])
            name_list.append(filename_detail)
            patch_idx_list.append(self.patch_idx)
            # (37, 1, 8, 8, 8)
            # (37, 1, 16, 16, 16)
            #print(np.shape(x_batch))
            #print(np.shape(y_batch))
        # print(np.array(x_batch).shape, np.array(x_batch).dtype)
        return np.array(x_batch), np.array(y_batch), name_list, patch_idx_list

    def on_epoch_end(self):
        # random shuffle
        #self.patch_lib_filenames = self.patch_lib_filenames
        #20210124
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




















def DefineTrainValSuperMudiDataloader_Stage2(gen_conf, train_conf, Para, is_shuffle_trainval = False):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    validation_split_ratio = train_conf['validation_split']
    path = dataset_info['path'][2]  # patch
    patch_dir = os.path.join(dataset_path, path)

    # get the path of the patch and the N_vol is 5
    #N_vol = len(sorted(glob.glob(patch_dir + '/*.npz')))
    N_vol = train_conf['num_training_patches']
    if is_shuffle_trainval is True:
        shuffle_order = np.random.permutation(N_vol)
    else:
        shuffle_order = np.arange(N_vol)

    # the num_training_patch is 40000, says there are 40000 patch seg out for training
    N = train_conf['num_training_patches']  # num of used training patches

    # the ratio 0.2 vol (1) for validation, and 0.8 vol (4) for training
    val_volumes = np.int32(np.ceil(N_vol * validation_split_ratio))
    train_volumes = N_vol - val_volumes

    # the ratio 0.2 patch (8000) for validation, and 0.8 patch (32000) for training
    N_val = np.int32(np.ceil(N * validation_split_ratio))
    N_train = N - N_val

    # shuffle_order is arrange, so no shuffle here, 32000 patch 
    trainDataloader = SuperMudiSequence_S2(gen_conf, train_conf, Para, shuffle_order[:train_volumes], N_train)
    if validation_split_ratio != 0:
        
        # shuffle_order is False, so 32000-4000 patch for val
        valDataloader = SuperMudiSequence_S2(gen_conf, train_conf, Para, shuffle_order[train_volumes:], N_val)
    else:
        valDataloader = None

    return trainDataloader, valDataloader



class SuperMudiSequence_S2(Sequence):
    def __init__(self, gen_conf, train_conf, Para, shuffle_order, N):
        dataset = train_conf['dataset']
        dataset_path = gen_conf['dataset_path']
        dataset_info = gen_conf['dataset_info'][dataset]
        self.para = Para
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
        '''
        self.patch_lib_filenames = np.array(sorted(glob.glob(self.patch_dir + '/*.npz')))
        self.patch_lib_filenames = self.patch_lib_filenames[shuffle_order]

        # # train-val split
        # self.patch_lib_filenames, self.N = self.__split_train_val(self.patch_lib_filenames, self.N, validation_split, is_val_gen)
        # expanded patch lib = N
        self.patch_lib_filenames = np.tile(self.patch_lib_filenames, math.ceil(self.N / len(self.patch_lib_filenames)))[:self.N]

        # random shuffle
        # 20210124
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)
        '''

        data_list = pd.read_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_label_train.csv')
        label_in = data_list[['Name', 'Level_1']]
        patch_lib_filenames_in = np.array(label_in['Name'])
        label_level_list_in = np.array(label_in['Level_1'])

        #print('!!!!!!!!!!!!!!!!!')
        #print(patch_lib_filenames_in[2:10])
        #print(label_level_list_in[2:10])

        self.patch_lib_filenames = patch_lib_filenames_in[shuffle_order]
        self.label_level_list = label_level_list_in[shuffle_order]

        #print('@@@@@@@@@@@@@@@@@')
        #print(self.patch_lib_filenames[2:10])
        #print(self.label_level_list[2:10])
        
        para_file=open('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/parameters.txt')
        txt=para_file.readlines()
        para_list=[]
        for w in txt:
            #w=w.replace('\n','')
            w=w.split()
            w = list(map(float, w))
            para_list.append(w)

        self.para_numpy = np.array(para_list)

        d_mean = np.mean(self.para_numpy, axis=0)
        d_std = np.std(self.para_numpy,axis=0)

        self.para_numpy = (self.para_numpy-d_mean)/d_std

        self.n_classes = 4

    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __getitem__(self, idx):

        batch_filenames = self.patch_lib_filenames[idx * self.batch_size:(idx+1)*self.batch_size]
        batch_label = self.label_level_list[idx * self.batch_size:(idx+1)*self.batch_size]

        #print('###########################')
        #print(batch_filenames)
        #print(batch_label)


        # batch[0] name is /scratch0/harrylin/2700564/patch/240-cdmri0014-(8, 8, 8)-(4, 4, 4)-0008.npz
        #print('batch name is,', batch_filenames[0])

        batch_error_label = []
        x_batch = []
        y_batch = []
        para_batch = []
        
        for filename in batch_filenames:
            #rnd_patch_idx += 1
            # filename is /scratch0/harrylin/2700564/patch/632-cdmri0011-(8, 8, 8)-(4, 4, 4)-0009.npz
            #print('filename is: ', filename)

            filename_detail = filename.split('-patch-')[0] + '.npz'
            x_patches, _ = self.__read_patch_data(filename_detail)

            specific_patch_num = filename.split('-patch-')[1]
            x_batch.append(x_patches[int(specific_patch_num)])

            volumn_num = filename.split('-cdmri')[0]
            volumn_num = volumn_num.split('/patch/')[1]
            para_info = self.para_numpy[int(volumn_num)]
            para_info = list(map(float, para_info))
            
            para_batch.append(para_info)

            #print(label_index.shape)
            #print(label_level)
            #file_dir = file_dir.split('.')[0]+'.png'
            # pick 1 out of the 10
            #print('idx,', rnd_patch_idx)

            
            # (37, 1, 8, 8, 8)
            # (37, 1, 16, 16, 16)
            #print(np.shape(x_batch))
            #print(np.shape(y_batch))


        #print(np.array(batch_filenames))
        #print(np.array(batch_label))

        if self.para=='para':
            return [np.array(x_batch), np.squeeze(np.array(para_batch))], to_categorical(np.array(batch_label), self.n_classes)
        else:
            return np.array(x_batch), to_categorical(np.array(batch_label), self.n_classes)

    def on_epoch_end(self):
        # random shuffle
        #self.patch_lib_filenames = self.patch_lib_filenames
        #20210124
        #if self.is_shuffle is True:
        #    np.random.shuffle(self.patch_lib_filenames)
        self.patch_lib_filenames = self.patch_lib_filenames

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







def DefineTrainValSuperMudiDataloader_Stage3_class(gen_conf, train_conf, para, is_shuffle_trainval = False):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    validation_split_ratio = train_conf['validation_split']

    
    path = dataset_info['path'][2]  # patch
    patch_dir = os.path.join(dataset_path, path)

    # get the path of the patch and the N_vol is 5
    #N_vol = len(sorted(glob.glob(patch_dir + '/*.npz')))
    N_vol = train_conf['num_training_patches']
    if is_shuffle_trainval is True:
        shuffle_order = np.random.permutation(N_vol)
    else:
        shuffle_order = np.arange(N_vol)

    # the num_training_patch is 40000, says there are 40000 patch seg out for training
    N = train_conf['num_training_patches']  # num of used training patches

    # the ratio 0.2 vol (1) for validation, and 0.8 vol (4) for training
    val_volumes = np.int32(np.ceil(N_vol * validation_split_ratio))
    train_volumes = N_vol - val_volumes

    # the ratio 0.2 patch (8000) for validation, and 0.8 patch (32000) for training
    N_val = np.int32(np.ceil(N * validation_split_ratio))
    N_train = N - N_val

    # shuffle_order is arrange, so no shuffle here, 32000 patch 
    trainDataloader = SuperMudiSequence_S3_class_label(gen_conf, train_conf, para, shuffle_order[:train_volumes], N_train)
    if validation_split_ratio != 0:
        
        # shuffle_order is False, so 32000-4000 patch for val
        valDataloader = SuperMudiSequence_S3_class_label(gen_conf, train_conf, para, shuffle_order[train_volumes:], N_val)
    else:
        valDataloader = None

    return trainDataloader, valDataloader



class SuperMudiSequence_S3_class_label(Sequence):
    def __init__(self, gen_conf, train_conf, para, shuffle_order, N):
        dataset = train_conf['dataset']
        dataset_path = gen_conf['dataset_path']
        dataset_info = gen_conf['dataset_info'][dataset]

        patch_shape = train_conf['patch_shape']
        extraction_step = train_conf['extraction_step']
        self.para = para
        #self.batch_size = train_conf['batch_size']
        self.batch_size = 1
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
        '''
        self.patch_lib_filenames = np.array(sorted(glob.glob(self.patch_dir + '/*.npz')))
        self.patch_lib_filenames = self.patch_lib_filenames[shuffle_order]

        # # train-val split
        # self.patch_lib_filenames, self.N = self.__split_train_val(self.patch_lib_filenames, self.N, validation_split, is_val_gen)
        # expanded patch lib = N
        self.patch_lib_filenames = np.tile(self.patch_lib_filenames, math.ceil(self.N / len(self.patch_lib_filenames)))[:self.N]

        # random shuffle
        # 20210124
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)
        '''

        data_list = pd.read_csv('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/Stage2_label_train.csv')
        label_in = data_list[['Name', 'Level_1']]
        patch_lib_filenames_in = np.array(label_in['Name'])
        label_level_list_in = np.array(label_in['Level_1'])

        #print('!!!!!!!!!!!!!!!!!')
        #print(patch_lib_filenames_in[2:10])
        #print(label_level_list_in[2:10])

        self.patch_lib_filenames = patch_lib_filenames_in[shuffle_order]
        self.label_level_list = label_level_list_in[shuffle_order]

        #print('@@@@@@@@@@@@@@@@@')
        #print(self.patch_lib_filenames[2:10])
        #print(self.label_level_list[2:10])
        
        para_file=open('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/parameters.txt')
        txt=para_file.readlines()
        para_list=[]
        for w in txt:
            #w=w.replace('\n','')
            w=w.split()
            w = list(map(float, w))
            para_list.append(w)

        self.para_numpy = np.array(para_list)

        d_mean = np.mean(self.para_numpy, axis=0)
        d_std = np.std(self.para_numpy,axis=0)

        self.para_numpy = (self.para_numpy-d_mean)/d_std

        self.n_classes = 4

    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __getitem__(self, idx):

        batch_filenames = self.patch_lib_filenames[idx * self.batch_size:(idx+1)*self.batch_size]
        batch_label = self.label_level_list[idx * self.batch_size:(idx+1)*self.batch_size]

        #print('###########################')
        #print(batch_filenames)
        #print(batch_label)


        # batch[0] name is /scratch0/harrylin/2700564/patch/240-cdmri0014-(8, 8, 8)-(4, 4, 4)-0008.npz
        #print('batch name is,', batch_filenames[0])

        batch_error_label = []
        x_batch = []
        y_batch = []
        para_batch = []
        
        for filename in batch_filenames:
            #rnd_patch_idx += 1
            # filename is /scratch0/harrylin/2700564/patch/632-cdmri0011-(8, 8, 8)-(4, 4, 4)-0009.npz
            #print('filename is: ', filename)

            filename_detail = filename.split('-patch-')[0] + '.npz'
            x_patches, _ = self.__read_patch_data(filename_detail)

            specific_patch_num = filename.split('-patch-')[1]
            x_batch.append(x_patches[int(specific_patch_num)])

            volumn_num = filename.split('-cdmri')[0]
            volumn_num = volumn_num.split('/patch/')[1]
            para_info = self.para_numpy[int(volumn_num)]
            para_info = list(map(float, para_info))
            
            para_batch.append(para_info)

        if self.para=='para':
            return [np.array(x_batch), np.array(para_batch)], np.array(batch_label), batch_filenames
        else:
            return np.array(x_batch), np.array(batch_label), batch_filenames

    def on_epoch_end(self):
        # random shuffle
        #self.patch_lib_filenames = self.patch_lib_filenames
        #20210124
        #if self.is_shuffle is True:
        #    np.random.shuffle(self.patch_lib_filenames)
        self.patch_lib_filenames = self.patch_lib_filenames

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





def DefineTrainValSuperMudiDataloader_Stage3(gen_conf, train_conf, csv_path, para='para', is_shuffle_trainval = False):
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]
    validation_split_ratio = train_conf['validation_split']

    path = dataset_info['path'][2]  # patch
    patch_dir = os.path.join(dataset_path, path)

    data_list = pd.read_csv(csv_path)
    #label_in = data_list[['Name']]
    patch_lib_filenames_in = np.array(data_list[['Name']])
    N_vol = len(patch_lib_filenames_in)

    # get the path of the patch and the N_vol is 5
    #N_vol = len(sorted(glob.glob(patch_dir + '/*.npz')))
    #N_vol = train_conf['num_training_patches']
    if is_shuffle_trainval is True:
        shuffle_order = np.random.permutation(N_vol)
    else:
        shuffle_order = np.arange(N_vol)

    # the num_training_patch is 40000, says there are 40000 patch seg out for training
    #N = train_conf['num_training_patches']  # num of used training patches
    N = len(patch_lib_filenames_in)
    # the ratio 0.2 vol (1) for validation, and 0.8 vol (4) for training
    val_volumes = np.int32(np.ceil(N_vol * validation_split_ratio))
    train_volumes = N_vol - val_volumes

    # the ratio 0.2 patch (8000) for validation, and 0.8 patch (32000) for training
    N_val = np.int32(np.ceil(N * validation_split_ratio))
    N_train = N - N_val

    # shuffle_order is arrange, so no shuffle here, 32000 patch 
    trainDataloader = SuperMudiSequence_S3(gen_conf, train_conf, para, shuffle_order[:train_volumes], N_train, csv_path)
    if validation_split_ratio != 0:
        
        # shuffle_order is False, so 32000-4000 patch for val
        valDataloader = SuperMudiSequence_S3(gen_conf, train_conf, para, shuffle_order[train_volumes:], N_val, csv_path)
    else:
        valDataloader = None

    return trainDataloader, valDataloader



class SuperMudiSequence_S3(Sequence):
    def __init__(self, gen_conf, train_conf, para, shuffle_order, N, csv_path):
        dataset = train_conf['dataset']
        dataset_path = gen_conf['dataset_path']
        dataset_info = gen_conf['dataset_info'][dataset]

        patch_shape = train_conf['patch_shape']
        extraction_step = train_conf['extraction_step']

        self.para = para
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
        '''
        self.patch_lib_filenames = np.array(sorted(glob.glob(self.patch_dir + '/*.npz')))
        self.patch_lib_filenames = self.patch_lib_filenames[shuffle_order]

        # # train-val split
        # self.patch_lib_filenames, self.N = self.__split_train_val(self.patch_lib_filenames, self.N, validation_split, is_val_gen)
        # expanded patch lib = N
        self.patch_lib_filenames = np.tile(self.patch_lib_filenames, math.ceil(self.N / len(self.patch_lib_filenames)))[:self.N]

        # random shuffle
        # 20210124
        if self.is_shuffle is True:
            np.random.shuffle(self.patch_lib_filenames)
        '''

        data_list = pd.read_csv(csv_path)
        
        patch_lib_filenames_in = np.array(data_list['Name'])

        #print('!!!!!!!!!!!!!!!!!')
        #print(patch_lib_filenames_in[2:10])
        #print(label_level_list_in[2:10])

        self.patch_lib_filenames = patch_lib_filenames_in[shuffle_order]

        #print('@@@@@@@@@@@@@@@@@')
        #print(self.patch_lib_filenames[2:10])
        #print(self.label_level_list[2:10])
        
        para_file=open('/cluster/project0/IQT_Nigeria/others/SuperMudi/code/iqt_supermudi-main/parameters.txt')
        txt=para_file.readlines()
        para_list=[]
        for w in txt:
            #w=w.replace('\n','')
            w=w.split()
            w = list(map(float, w))
            para_list.append(w)

        self.para_numpy = np.array(para_list)

        d_mean = np.mean(self.para_numpy, axis=0)
        d_std = np.std(self.para_numpy,axis=0)

        self.para_numpy = (self.para_numpy-d_mean)/d_std

        self.n_classes = 4

    def __len__(self):
        return math.ceil(self.N / self.batch_size)

    def __getitem__(self, idx):

        batch_filenames = self.patch_lib_filenames[idx * self.batch_size:(idx+1)*self.batch_size]

        #print('###########################')
        #print(batch_filenames)
        #print(batch_label)


        # batch[0] name is /scratch0/harrylin/2700564/patch/240-cdmri0014-(8, 8, 8)-(4, 4, 4)-0008.npz
        #print('batch name is,', batch_filenames[0])

        batch_error_label = []
        x_batch = []
        y_batch = []
        para_batch = []
        
        for filename in batch_filenames:
            #rnd_patch_idx += 1
            # filename is /scratch0/harrylin/2700564/patch/632-cdmri0011-(8, 8, 8)-(4, 4, 4)-0009.npz
            #print('filename is: ', filename)

            filename_detail = filename.split('-patch-')[0] + '.npz'
            x_patches, y_patches = self.__read_patch_data(filename_detail)

            specific_patch_num = filename.split('-patch-')[1]
            x_batch.append(x_patches[int(specific_patch_num)])
            y_batch.append(y_patches[int(specific_patch_num)])


        #print(np.array(batch_filenames))
        #print(np.array(batch_label))

        if self.para=='para':
            return [np.array(x_batch), np.squeeze(np.array(para_batch))], np.array(y_batch)
        else:
            return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        # random shuffle
        #self.patch_lib_filenames = self.patch_lib_filenames
        #20210124
        #if self.is_shuffle is True:
        #    np.random.shuffle(self.patch_lib_filenames)
        self.patch_lib_filenames = self.patch_lib_filenames

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