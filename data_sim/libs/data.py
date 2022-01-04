import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tensorflow import keras

sys.path.append("libs")
from utils import sigmoid


class CollaborativeVAEDataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 data_root,
                 phase,
                 batch_size,
                 simulate,
                 reuse=True,
                 joint=False,
                 shuffle=True):
        '''
            Generate the training and validation data
        '''
        assert phase in ["train", "val", "test"]
        assert simulate in ["exposure", "ratings"]
        self.phase = phase
        self.simulate = simulate
        self.batch_size = batch_size

        self.__load_data(data_root, reuse=reuse)

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

    def __load_data(self, data_root, reuse):
        ### Load the dataset
        meta_table = pd.read_csv(os.path.join(data_root, "meta.csv"))
        self.num_items = meta_table["num_items"][0]

        if self.phase == "train":
            obs_path = os.path.join(data_root, "train.csv")
            obs_records = pd.read_csv(obs_path)
            obs_group = obs_records.groupby("uid")
            unk_group = obs_group
        else:
            obs_path = os.path.join(data_root, "{}_obs.csv".format(self.phase))
            unk_path = os.path.join(data_root, "{}_unk.csv".format(self.phase))
            obs_records = pd.read_csv(obs_path)
            unk_records = pd.read_csv(unk_path)
            obs_group = obs_records.groupby("uid")
            unk_group = unk_records.groupby("uid")

        ### IDs and corresponding indexes
        self.user_ids = np.array(pd.unique(obs_records["uid"]), dtype=np.int32)
        self.indexes = np.arange(len(self.user_ids))
        self.num_users = len(self.user_ids)

        X_path = os.path.join(data_root, "{}_X.npz".format(self.phase))
        Y_path = os.path.join(data_root, "{}_Y.npz".format(self.phase))

        if reuse and os.path.exists(X_path) and os.path.exists(Y_path):
            self.X = sparse.load_npz(X_path)
            self.Y = sparse.load_npz(Y_path)
        else:
            ### Represent the whole dataset with a huge sparse matrix
            rows_X, cols_X, rows_Y, cols_Y = [], [], [], []
            if self.simulate == "ratings":
                ratings_X, ratings_Y = [], []
                
            for i, user_id in enumerate(self.user_ids):
                group_X = obs_group.get_group(user_id)
                group_Y = unk_group.get_group(user_id)
                rows_X += [i]*len(group_X); cols_X += list(group_X["vid"]-1)
                rows_Y += [i]*len(group_Y); cols_Y += list(group_Y["vid"]-1)                

                if self.simulate == "ratings":
                    ratings_X += list(group_X["rating"])
                    ratings_Y += list(group_Y["rating"])
            
            if self.simulate == "exposure":
                ratings_X = np.ones_like(rows_X, dtype=np.float32)
                ratings_Y = np.ones_like(rows_Y, dtype=np.float32)
            
            self.X = sparse.csr_matrix((ratings_X,(rows_X, cols_X)), dtype='float32',
                                       shape=(self.num_users, self.num_items))
            
            self.Y = sparse.csr_matrix((ratings_Y,(rows_Y, cols_Y)), dtype='float32',
                                       shape=(self.num_users, self.num_items))

            if reuse:
                sparse.save_npz(X_path, self.X)
                sparse.save_npz(Y_path, self.Y)

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
        batch_X = self.X[batch_idxes].toarray() 
        batch_Y = self.Y[batch_idxes].toarray() 
        return (batch_X, batch_Y)

    @property
    def target_shape(self):
        return self._target_shape
    

if __name__ == '__main__':
    pass