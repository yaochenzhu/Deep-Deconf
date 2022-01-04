import os
import time
import logging
import argparse

import sys
sys.path.append("libs")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from data import CollaborativeVAEDataGenerator
from train import get_collabo_vae

from evaluate import EvaluateModel
from evaluate import Recall_at_k, NDCG_at_k
from evaluate import Recall_at_k_explicit, NDCG_at_k_explicit


def predict_and_evaluate():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="specify the dataset")
    parser.add_argument("--split", type=int, default=0,
        help="specify the split of dataset for experiment")
    parser.add_argument("--batch_size", type=int, default=500,
        help="specify the batch size for prediction")
    parser.add_argument("--device" , type=str, default="0",
        help="specify the visible GPU device")
    parser.add_argument("--simulate", type=str, choices=["exposure", "ratings"],
        help="to simulate the exposure or the ratings")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Fix the random seeds.
    np.random.seed(98765)
    tf.set_random_seed(98765)

    ### Get the test data generator for content vae
    data_root = os.path.join("data", args.dataset, str(args.split), args.simulate)
    model_root = os.path.join("models", args.dataset, str(args.split), args.simulate)

    test_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase = "test", 
        batch_size = args.batch_size, joint=True,
        shuffle=False, simulate=args.simulate
    )

    ### Build test model and load trained weights
    collab_vae = get_collabo_vae(args.dataset, test_gen.num_items)
    collab_vae.load_weights(os.path.join(model_root, "best.model"))

    vae_eval = collab_vae.build_vae_eval()

    if args.simulate == "exposure":
        recall_func = Recall_at_k; NDCG_func = NDCG_at_k
    elif args.simulate == "ratings":
        recall_func = Recall_at_k_explicit; NDCG_func = NDCG_at_k_explicit

    ### Evaluate and save the results
    k4recalls = [10]
    k4ndcgs = [10]
    recalls, NDCGs = [], []
    for k in k4recalls:
        recalls.append("{:.4f}".format(EvaluateModel(vae_eval, test_gen, recall_func, k=k)))
    for k in k4ndcgs:
        NDCGs.append("{:.4f}".format(EvaluateModel(vae_eval, test_gen, NDCG_func, k=k)))

    recall_table = pd.DataFrame({"k":k4recalls, "recalls":recalls}, columns=["k", "recalls"])
    recall_table.to_csv(os.path.join(model_root, "recalls.csv"), index=False)

    ndcg_table = pd.DataFrame({"k":k4ndcgs, "NDCGs": NDCGs}, columns=["k", "NDCGs"])
    ndcg_table.to_csv(os.path.join(model_root, "NDCGs.csv"), index=False)

    print("Done evaluation! Results saved to {}".format(model_root))


if __name__ == '__main__':
    predict_and_evaluate()