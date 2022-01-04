import os
import pdb
import time
import logging
import argparse

import sys
sys.path.append(os.path.join("libs", "exposure"))

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import sparse
from tensorflow.keras import backend as K

from data import ExposureVAEDataGenerator
from train_exposure import get_collabo_vae

### Fix the random seeds.
np.random.seed(98765)
tf.set_random_seed(98765)


def predict_and_evaluate():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="specify the dataset")
    parser.add_argument("--split", type=int, default=0,
        help="specify the split of dataset for experiment")
    parser.add_argument("--conf", type=float, default=0,
        help="specify the confounding effects")
    parser.add_argument("--batch_size", type=int, default=500,
        help="specify the batch size for prediction")
    parser.add_argument("--device" , type=str, default="0",
        help="specify the visible GPU device")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Get the test data generator for content vae
    data_root = os.path.join("data", args.dataset, str(args.split), "{:.1f}".format(args.conf))
    model_root = os.path.join("models", args.dataset, str(args.split), "{:.1f}".format(args.conf))

    ### Load train, val, test dataset
    train_X = sparse.load_npz(os.path.join(data_root, "train_obs_rat.npz")).A
    val_X = (sparse.load_npz(os.path.join(data_root, "val_obs_rat.npz")) + \
             sparse.load_npz(os.path.join(data_root, "val_hdt_rat.npz"))).A
    test_X = sparse.load_npz(os.path.join(data_root, "test_obs_rat.npz")).A

    train_X = (train_X > 0).astype(np.int32)
    val_X = (val_X > 0).astype(np.int32)
    test_X = (test_X > 0).astype(np.int32)

    ### Build test model and load trained weights
    collab_vae = get_collabo_vae(args.dataset, train_X.shape[1])
    collab_vae.load_weights(os.path.join(model_root, "best_exposure.model"))

    vae_infer = collab_vae.build_vae_infer()

    train_Z = vae_infer.predict_on_batch(train_X) 
    val_Z = vae_infer.predict_on_batch(val_X)
    test_Z = vae_infer.predict_on_batch(test_X)

    save_root = os.path.join(data_root, "exposure")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    names = ["train_subs_conf.npy", "val_subs_conf.npy", "test_subs_conf.npy"]
    values = [train_Z, val_Z, test_Z]

    for name, value in zip(names, values):
        np.save(os.path.join(save_root, name), value)

    print("Done inference for substitute confounders!")

if __name__ == '__main__':
    predict_and_evaluate()