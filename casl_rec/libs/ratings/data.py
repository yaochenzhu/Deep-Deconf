import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tensorflow import keras

sys.path.append("libs")
from utils import sigmoid


class RatingOutcomeModelGenerator(keras.utils.Sequence):
    '''
        Generate the training and validation data
    '''
    def __init__(self, 
                 data_root,
                 phase,
                 batch_size,
                 use_feature=True,
                 use_exposure=True,
                 shuffle=True):
        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.batch_size = batch_size
        self.use_feature = use_feature
        self.use_exposure = use_exposure

        self.__load_data(data_root)

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

    def __load_data(self, data_root):
        ### Load the dataset
        exposure_root = os.path.join(data_root, "exposure")
        self.subs_conf = np.load(os.path.join(exposure_root, "{}_subs_conf.npy".format(self.phase)))
        self.conf_dim = self.subs_conf.shape[-1]
        names, dims = ["subs_conf"], [self.conf_dim]
        ### Use observed exposure as surrogate?
        if self.use_exposure:
            self.exposure = sparse.load_npz(os.path.join(data_root, "{}_obs_rat.npz".format(self.phase))).A
            self.exposure = (self.exposure>0).astype(np.int32)
            self.exposure_dim = self.exposure.shape[-1]
            names.append("exposure"); dims.append(self.exposure_dim)
        ### Use user features to reduce variance?
        if self.use_feature:
            self.features = np.load(os.path.join(data_root, "{}_feat.npy".format(self.phase)))
            self.feature_dim = self.features.shape[-1]
            names.append("features"); dims.append(self.feature_dim)
        ### When training, we use all the OBSERVED ratings.
        if self.phase == "train":
            self.ratings = sparse.load_npz(os.path.join(data_root, "train_obs_rat.npz")).A
        ### When validation, 20% observations are HOLD-OUT for model selection.
        elif self.phase == "val":
            self.ratings = sparse.load_npz(os.path.join(data_root, "val_hdt_rat.npz")).A
        ### When testing, we have ALL THE RATINGS to calculate the unbiased metrics.
        else:
            self.ratings = np.load(os.path.join(data_root, "test_unk_rat.npy"))
        self.num_users, self.num_items = self.ratings.shape
        self.indexes = np.arange(self.num_users)
        names.append("ratings"); dims.append(self.num_items)
        self._name_dim_dict = dict(zip(names, dims))

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
        batch_X = [self.subs_conf[batch_idxes]]
        if self.use_exposure:
            batch_X += [self.exposure[batch_idxes]]
        if self.use_feature:
            batch_X += [self.features[batch_idxes]]
        batch_Y = self.ratings[batch_idxes]
        return (batch_X, batch_Y)

    @property
    def name_dim_dict(self):
        return self._name_dim_dict
    

if __name__ == '__main__':
    pass