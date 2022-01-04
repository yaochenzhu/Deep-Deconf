import os
import time
import logging
import argparse

import sys
sys.path.append(os.path.join("libs", "ratings"))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from data import RatingOutcomeModelGenerator
from train_ratings import get_collabo_vae

from evaluate import EvaluateModel
from evaluate import Recall_at_k, NDCG_at_k
from evaluate import Recall_at_k_explicit, NDCG_at_k_explicit

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

    test_gen = RatingOutcomeModelGenerator(
        data_root = data_root, phase = "test", 
        batch_size = args.batch_size, shuffle=False, 
    )

    ### Build test model and load trained weights
    collab_vae = get_collabo_vae(args.dataset, test_gen.name_dim_dict)
    collab_vae.load_weights(os.path.join(model_root, "best_ratings.model"))

    _, vae_eval = collab_vae.build_outcome_model()
    recall_func = Recall_at_k_explicit; NDCG_func = NDCG_at_k_explicit

    ### Evaluate and save the results
    k4recalls = [5, 10, 20, 30, 40, test_gen.num_items-1]
    k4ndcgs = [5, 10, 20, 30, 40, test_gen.num_items-1]
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