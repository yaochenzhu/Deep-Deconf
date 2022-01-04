import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
from scipy import sparse

import warnings
warnings.filterwarnings('ignore')

def get_counts(raw_data, attr):
    counts_group = raw_data[[attr]].groupby(attr, as_index=False)
    counts = counts_group.size()
    return counts


def filter_triplets_exposure(raw_data, min_ucount, min_icount):
    new_data = raw_data[raw_data['rating'] > 0]
    for i in range(3):
        if min_icount > 0:
            i_counts = get_counts(new_data, 'vid')
            new_data = new_data[new_data['vid'].isin(i_counts.index[i_counts >= min_icount])]
        if min_ucount > 0:
            u_counts = get_counts(new_data, 'uid')
            new_data = new_data[new_data['uid'].isin(u_counts.index[u_counts >= min_ucount])]
    return new_data


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


if __name__ == '__main__':
    '''
        Basic usage:
            python prepare_data.py --dataset ml-1m --simulate exposure
            python prepare_data.py --dataset ml-1m --simulate ratings
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="specify the dataset")
    parser.add_argument("--source", type=str, choices=["user"],
        default="user", help="use vae to embed users or items")
    parser.add_argument("--min_ucount", type=int, default=5,
        help="keep users who clicked on at least min_ucount items")
    parser.add_argument("--min_icount", type=int, default=1,
        help="keep items which were clicked on by at least min_icount users")
    parser.add_argument("--num_splits", type=int, default=10,
        help="split the data into num_splits folds")
    parser.add_argument("--test_percent", type=float, default=0.05,
        help="use which percent of user as test user")
    parser.add_argument("--unk_percent", type=float, default=0.2,
        help="for each user, use which percent of ratings as observed ratings")
    parser.add_argument("--simulate", type=str, choices=["exposure", "ratings"],
        help="to simulate the exposure or the ratings")
    args = parser.parse_args()

    random.seed(98765)
    np.random.seed(98765)

    source = "uid" if args.source == "user" else "vid"
    target = "vid" if args.source == "user" else "uid"
    args.target = "item" if args.source == "user" else "user"

    ### Read raw rating data
    raw_data = pd.read_csv(os.path.join(args.dataset, 'ratings.dat'), sep="::", header=None)
    raw_data = raw_data[[0, 1, 2]].rename(columns={0:"uid", 1:"vid", 2:"rating"})

    ### Process the raw data
    filtered_data = filter_triplets_exposure(raw_data, args.min_ucount, args.min_icount)
    filtered_data = filtered_data[["uid", "vid", "rating"]]
    if args.simulate == "ratings":
        '''
            To keep the items in ratings compatible to that in exposure
        '''
        remaining_vids = filtered_data.vid.unique()
        filtered_data = raw_data[raw_data.vid.isin(remaining_vids)]
    else:
        filtered_data = filtered_data[["uid", "vid"]]

    unique_sids = filtered_data.groupby(source).size().index
    unique_tids = filtered_data.groupby(target).size().index
    source_old2new = {sid:(i+1) for (i, sid) in enumerate(np.sort(unique_sids))}
    target_old2new = {tid:(i+1) for (i, tid) in enumerate(np.sort(unique_tids))}

    num_users = len(unique_sids)
    num_items = len(unique_tids)
    num_ratings = len(filtered_data)
    sparsity = num_ratings / (num_users * num_items)
    ucg = filtered_data.groupby("uid").count()
    max_u, min_u, avg_u, std_u = ucg.max()[0], ucg.min()[0], ucg.mean()[0], ucg.std()[0]
    vcg = filtered_data.groupby("vid").count()
    max_v, min_v, avg_v, std_v = vcg.max()[0], vcg.min()[0], vcg.mean()[0], vcg.std()[0]

    save_root = os.path.join("..", "data", args.dataset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    with open(os.path.join(save_root, "meta.csv"), "w") as f:
        f.write("num_user,{}\n".format(num_users))
        f.write("num_item,{}\n".format(num_items))
        f.write("sparsity,{:.3f}%\n".format(sparsity*100))
        f.write("max user interations,{}\n".format(max_u))
        f.write("min user interations,{}\n".format(min_u))
        f.write("avg user interations,{:.3f}\n".format(avg_u))
        f.write("std user interations,{:.3f}\n".format(std_u))
        f.write("max item interations,{}\n".format(max_v))
        f.write("min item interations,{}\n".format(min_v))
        f.write("avg item interations,{:.3f}\n".format(avg_v))
        f.write("std item interations,{:.3f}\n".format(std_v))

    meta_table = pd.DataFrame({"num_{}s".format(args.source):[len(unique_sids)], 
                               "num_{}s".format(args.target):[len(unique_tids)]})


    for i in range(args.num_splits):
        print("Begin the {}th data split!".format(i+1))

        idxes_perm = np.random.permutation(unique_sids.size)
        num_test  = int(args.test_percent*unique_sids.size)
        train_end = int(unique_sids.size - 2*num_test)
        val_end   = train_end + num_test

        train_ids = unique_sids[idxes_perm[:train_end]]
        val_ids   = unique_sids[idxes_perm[train_end:val_end]]
        test_ids  = unique_sids[idxes_perm[val_end:]]

        train_data = filtered_data[filtered_data[source].isin(train_ids)]
        val_data   = filtered_data[filtered_data[source].isin(val_ids)]
        test_data  = filtered_data[filtered_data[source].isin(test_ids)]

        val_obs_data , val_unk_data  = split_observed_unknown(val_data,  source, unk_frac=args.unk_percent)
        test_obs_data, test_unk_data = split_observed_unknown(test_data, source, unk_frac=args.unk_percent)

        tables = [train_data, val_obs_data, val_unk_data, test_obs_data, test_unk_data]
        for data in tables:
            data.sort_values(by=[source, target], inplace=True)
            data[source] = data[source].apply(lambda x:source_old2new[x])

        save_subroot = os.path.join(save_root, str(i), args.simulate)
        if not os.path.exists(save_subroot):
            os.makedirs(save_subroot)

        names = ["train.csv", "val_obs.csv", "val_unk.csv", "test_obs.csv", "test_unk.csv"]
        for name, data in zip(names, tables):
            data[target] = data[target].apply(lambda x:target_old2new[x])
            data.to_csv(os.path.join(save_subroot, name), index=False)
        meta_table.to_csv(os.path.join(save_subroot, "meta.csv"), index=False)