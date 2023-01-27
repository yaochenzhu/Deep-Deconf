import os
import sys
import glob
import random
import argparse

import numpy as np
import pandas as pd
from scipy import sparse

import warnings
warnings.filterwarnings('ignore')

random.seed(98765)
np.random.seed(98765)

def get_counts(raw_data, attr):
    counts_group = raw_data[[attr]].groupby(attr, as_index=False)
    counts = counts_group.size()
    return counts


def split_observed_unknown(data, source, unk_frac=0.2):
    data_group = data.groupby(source)
    obs_list, unk_list = [], []
    for i, (_, group) in enumerate(data_group):
        n_records = len(group)
        if n_records >= 5:
            idx = np.zeros(n_records, dtype='bool')
            idx[np.random.choice(n_records, size=max(int(unk_frac*n_records), 1), 
                                 replace=False).astype('int64')] = True
            obs_list.append(group[np.logical_not(idx)])
            unk_list.append(group[idx])
        else:
            obs_list.append(group)
        if i % 200 == 0:
            print("{} source sampled".format(i))
            sys.stdout.flush()
    data_obs = pd.concat(obs_list)
    data_unk = pd.concat(unk_list)
    return data_obs, data_unk


def split_validation_ratings(ratings, holdout_rate, shape):
    ratings = sparse.coo_matrix(ratings)
    rating_table = pd.DataFrame(
        {"uid":ratings.row, "vid":ratings.col, "ratings":ratings.data},
        columns=["uid", "vid", "ratings"]
        )
    val_obs_table, val_hdt_table = split_observed_unknown(rating_table, "uid", holdout_rate)

    val_obs_ratings = sparse.csr_matrix((list(val_obs_table.ratings),
        (list(val_obs_table.uid), list(val_obs_table.vid))), shape=shape)
    val_hdt_ratings = sparse.csr_matrix((list(val_hdt_table.ratings),
        (list(val_hdt_table.uid), list(val_hdt_table.vid))), shape=shape)
    return val_obs_ratings, val_hdt_ratings


if __name__ == '__main__':
    '''
        Usage:
            python preprocess --dataset ml-1m
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="specify the dataset")
    parser.add_argument("--test_rate", type=float, default=0.10,
        help="use which percent of user as test user")
    parser.add_argument("--holdout_rate", type=float, default=0.2,
        help="use which percent of exposure for prediction check")
    parser.add_argument("--split", type=int, default=5,
        help="split the dataset into which number of train/val/test")
    args = parser.parse_args()

    roots = glob.glob(os.path.join(args.dataset, "0", "*"))
    roots = [root for root in roots if os.path.exists(os.path.join(root, "raw"))]

    for root in roots:
        raw_root = os.path.join(root, "raw")
        raw_exposure = np.load(os.path.join(raw_root, "exposure.npy"))
        raw_ratings = np.load(os.path.join(raw_root, "ratings.npy"))
        raw_features = np.load(os.path.join(raw_root, "user_features.npy"))

        obs_ratings = raw_ratings*raw_exposure
        unk_ratings = raw_ratings*(1 - raw_exposure)
        num_users, num_items = obs_ratings.shape

        for split in range(args.split):
            idxes_perm = np.random.permutation(num_users)
            num_test  = int(args.test_rate*num_users)
            train_end = int(num_users - 2*num_test)
            val_end   = train_end + num_test
            assert train_end > 0, "the number of training samples should > 0"

            train_idxes = idxes_perm[:train_end]
            val_idxes = idxes_perm[train_end:val_end]
            test_idxes = idxes_perm[val_end:]

            train_obs_ratings = sparse.csr_matrix(obs_ratings[train_idxes])
            val_obs_ratings = sparse.csr_matrix(obs_ratings[val_idxes])
            test_obs_ratings = sparse.csr_matrix(obs_ratings[test_idxes])

            val_unk_ratings = unk_ratings[val_idxes]
            test_unk_ratings = unk_ratings[test_idxes]

            train_features = raw_features[train_idxes]
            val_features = raw_features[val_idxes]
            test_features = raw_features[test_idxes]

            val_obs_ratings, val_hdt_ratings = split_validation_ratings(\
                val_obs_ratings, args.holdout_rate, shape=[num_test, num_items])

            ratings = [train_obs_ratings, val_obs_ratings, val_hdt_ratings,
                       val_unk_ratings, test_obs_ratings, test_unk_ratings]
            names = ["train_obs_rat.npz", "val_obs_rat.npz", "val_hdt_rat.npz",
                     "val_unk_rat.npy", "test_obs_rat.npz", "test_unk_rat.npy"]

            save_root = root.split(os.path.sep)
            save_root[1] = str(split)
            save_root = os.path.sep.join(save_root)
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            for (name, rating) in zip(names, ratings):
                if name.endswith(".npy"):
                    np.save(os.path.join(save_root, name), rating)
                elif name.endswith(".npz"):
                    sparse.save_npz(os.path.join(save_root, name), rating)
                else:
                    raise NotImplementedError

            features = [train_features, val_features, test_features]
            names = ["train_feat.npy", "val_feat.npy", "test_feat.npy"]

            for (name, feature) in zip(names, features):
                np.save(os.path.join(save_root, name), feature)

            print("Done for {}".format(save_root))
    print("Done preprocessing the {} data!".format(args.dataset))
