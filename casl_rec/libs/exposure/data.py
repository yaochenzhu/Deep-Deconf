import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tensorflow import keras

sys.path.append("libs")
from utils import sigmoid


class ExposureVAEDataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 data_root,
                 phase,
                 batch_size,
                 shuffle=True):
        '''
            Generate the training and validation data
        '''
        assert phase in ["train", "val"]
        self.phase = phase
        self.batch_size = batch_size

        self.__load_data(data_root)

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

    def __load_data(self, data_root):
        ### Load the dataset
        if self.phase == "train":
            ### For training 
            self.X = sparse.load_npz(os.path.join(data_root, "train_obs_rat.npz"))
            self.Y = self.X.copy()
        elif self.phase == "val":
            ### For prediction check (select model) 
            self.X = sparse.load_npz(os.path.join(data_root, "val_obs_rat.npz"))
            self.Y = sparse.load_npz(os.path.join(data_root, "val_hdt_rat.npz"))
        self.X = self.__binarize(self.X)
        self.Y = self.__binarize(self.Y)
        self.num_users, self.num_items = self.X.shape
        self.indexes = np.arange(self.num_users)

    def __binarize(self, ratings):
        return (ratings > 0).astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        batch_num = self.num_users//self.batch_size
        if self.num_users%self.batch_size != 0:
            batch_num+=1
        return batch_num

    def __getitem__(self, i):
        batch_idxes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_X = self.X[batch_idxes].A
        batch_Y = self.Y[batch_idxes].A
        return (batch_X, batch_Y)

    @property
    def target_shape(self):
        return self.num_items
    

if __name__ == '__main__':
    pass