import os
import pdb
import time
import logging
import argparse

import sys
sys.path.append(os.path.join("libs", "exposure"))
from utils import Init_logging
from utils import PiecewiseSchedule

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from layers import AddGaussianLoss
from layers import AddBernoulliLoss

from data import ExposureVAEDataGenerator
from model import ExposureVariationalAutoencoder
from evaluate import binary_crossentropy

import warnings
warnings.filterwarnings('ignore')

### Fix the random seeds.
np.random.seed(98765)
tf.set_random_seed(98765)

simulate_args = {
    "hidden_sizes":[], 
    "latent_size":100,
    "encoder_activs" : [],
    "decoder_activs" : ["sigmoid"],
    "dropout_rate" : 0.5
}

movielen_args = {
    "hidden_sizes":[], 
    "latent_size":100,
    "encoder_activs" : [],
    "decoder_activs" : ["sigmoid"],
    "dropout_rate" : 0.5
}

amazon_args = {
    "hidden_sizes":[], 
    "latent_size":100,
    "encoder_activs" : [],
    "decoder_activs" : ["sigmoid"],
    "dropout_rate" : 0.5
}

data_args_dict = {
    "simulate" : simulate_args,
    "ml-1m" : movielen_args,
    "amazon-vg" : amazon_args
}


def get_collabo_vae(dataset, input_dim):
    get_collabo_vae = ExposureVariationalAutoencoder(
         input_dim = input_dim,
         **data_args_dict[dataset]
    )
    return get_collabo_vae


def train_vae_model():
    '''
        Basic usage:
            python train_exposure.py --dataset ml-1m --split 0 --conf 0
    '''
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="specify the dataset")
    parser.add_argument("--split", type=int, default=0,
        help="specify the split of dataset for experiment")
    parser.add_argument("--conf", type=float, default=0,
        help="specify the confounding effects")
    parser.add_argument("--batch_size", type=int, default=500,
        help="specify the batch size for updating the vae")
    parser.add_argument("--device" , type=str, default="0",
        help="specify the visible GPU device")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Get the train, val data generator for exposure vae
    data_root = os.path.join("data", args.dataset, str(args.split), "{:.1f}".format(args.conf))
    train_gen = ExposureVAEDataGenerator(
        data_root = data_root, phase="train",
        batch_size = args.batch_size,
    )
    valid_gen = ExposureVAEDataGenerator(
        data_root = data_root, phase="val",
        batch_size = args.batch_size*8,
    )

    ### Some configurations for training
    lr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [51, 5e-4]], outside_value=5e-4)   
    collabo_vae = get_collabo_vae(args.dataset, input_dim=train_gen.num_items)
    vae_train = collabo_vae.build_vae_train()
    vae_eval = collabo_vae.build_vae_eval()

    best_neg_loglld = np.inf

    save_root = os.path.join("models", args.dataset, str(args.split), "{:.1f}".format(args.conf))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    training_dynamics = os.path.join(save_root, "train_dynamic_exp.csv")
    with open(training_dynamics, "w") as f:
        f.write("train_neg_loglld,val_neg_loglld,\n")
    best_path = os.path.join(save_root, "best_exposure.model")

    lamb_schedule_gauss = PiecewiseSchedule([[0, 0.0], [80, 0.2]], outside_value=0.2)
    vae_train.compile(loss=binary_crossentropy, optimizer=optimizers.Adam(), metrics=[binary_crossentropy])

    epochs = 150
    for epoch in range(epochs):
        ### Set the value of annealing parameters
        K.set_value(vae_train.optimizer.lr, lr_schedule.value(epoch))
        K.set_value(collabo_vae.add_gauss_loss.lamb_kl, lamb_schedule_gauss.value(epoch))
        print("-"*10 + "Epoch:{}".format(epoch), "-"*10)

        stats = vae_train.fit_generator(train_gen, workers=4, epochs=1, validation_data=valid_gen)
        train_neg_loglld = stats.history["binary_crossentropy"][0]
        valid_neg_loglld = stats.history["val_binary_crossentropy"][0]

        if valid_neg_loglld < best_neg_loglld:
            best_neg_loglld = valid_neg_loglld
            vae_train.save_weights(best_path, save_format="tf")

        with open(training_dynamics, "a") as f:
            f.write("{:.4f},{:.4f}".format(train_neg_loglld, valid_neg_loglld))

        print("-"*5+"Epoch: {}".format(epoch)+"-"*5)
        print("cur neg_loglld: {:.4f}, best neg_loglld: {:.4f}".format(valid_neg_loglld, best_neg_loglld))

if __name__ == '__main__':
    train_vae_model()