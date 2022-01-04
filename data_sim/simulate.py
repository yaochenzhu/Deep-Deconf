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
from sklearn.decomposition import PCA

from train import get_collabo_vae
from data import CollaborativeVAEDataGenerator

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

title_font = {
    'family' : "Arial",
    'weight' : 'normal',
    'size'   : 10.5,
}

label_font = {
    'family' : "Arial",
    'weight' : 'normal',
    'size'   : 9,    
}

plt.style.use("bmh")


def get_exp_rate(exp_table):
    num_users = exp_table.uid.unique().size
    num_items = exp_table.uid.unique().size
    num_interactions = exp_table.size
    return num_interactions/(num_users*num_items)


def adj_rat_dist(rat_dist_raw, weighted=True):
    if weighted:
        base=2.4
        weights = np.array([base**i for i in range(5)][::-1])
    else:
        weights = np.array([1,1,1,1,1])
    rat_dist_raw = rat_dist_raw*weights
    return rat_dist_raw / rat_dist_raw.sum()


def get_rat_dist(rat_table):
    rat_dist_raw = np.array(
        rat_table.groupby("rating").count()["uid"].sort_index())
    return adj_rat_dist(rat_dist_raw)


def plot_exp_stats(exp_sim, save_root, fmt, name="exp_stats", title_="default"):
    item_pop = exp_sim.sum(axis=0)
    pop_table = pd.DataFrame({"pop":item_pop})
    fig, ax = plt.subplots(figsize=(3.50, 2.00))
    title = "Item Exposure Distribution" if title_ == "default" else title_
    ax.set_title(title, title_font)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    [label.set_fontsize(8) for label in labels]

    pop_table.plot(kind="hist", ax=ax, density=True, bins=15, alpha=0.65, legend=False)
    pop_table.plot(kind="kde", ax=ax, legend=False)
    max_x = int(item_pop.max()//50)*50; ax.set_xlim(0, max_x)
    ax.set_xlabel("#Interactions", label_font)
    ax.set_yticks([]); ax.set_ylabel("")
    ax.tick_params(left = False, bottom = False)

    for _, spine in ax.spines.items():
        spine.set_visible(False)
    ax.grid(False)

    qt_25, qt_50, qt_75 = pop_table.quantile(0.25), pop_table.quantile(0.5), pop_table.quantile(0.75)
    qts = [[qt_25, 0.8, 0.26], [qt_50, 1, 0.36],  [qt_75, 0.8, 0.46]]
    for qt in qts:
        ax.axvline(qt[0][0], alpha=qt[1], ymax=qt[2], linestyle=":")

    ax.text(qt_25[0]/max_x+0.01, 0.26, "25th", size=5.85, alpha=0.85,
        verticalalignment='center', transform=ax.transAxes)
    ax.text(qt_50[0]/max_x+0.01, 0.36, "50th", size=6.25, alpha=1,
        verticalalignment='center', transform=ax.transAxes)
    ax.text(qt_75[0]/max_x+0.01, 0.46, "75th Percentile", size=5.85, alpha=0.85,
        verticalalignment='center', transform=ax.transAxes)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    fig.savefig(os.path.join(save_root, "{}.{}".format(name, fmt)), bbox_inches='tight')


def plot_rat_stats(rat_sim, save_root, fmt):
    for rating in range(1, 6):
        exp_sim = (rat_sim == rating).astype(np.int32)
        name="rat_stats_{}.{}".format(rating, fmt)
        title="Item Rating {} Distribution".format(rating)
        plot_exp_stats(exp_sim, save_root, fmt, name, title)


def plot_confounding_effects(exp_sim, rat_sim, conf_coeff, save_root, fmt,):
    rat_true = rat_sim
    rat_obsv = rat_sim*exp_sim
    fig, ax = plt.subplots(figsize=(3.50, 2.00))
    title = "Confounding Effects for {:.1f}.".format(conf_coeff)
    ax.set_title(title, title_font)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    [label.set_fontsize(8) for label in labels]

    count_true, count_obsv = [], []
    for i in range(1, 6):
        count_true.append((rat_true == i).sum())
        count_obsv.append((rat_obsv == i).sum())
    dist_true = np.array(count_true) / np.sum(count_true)
    dist_obsv = np.array(count_obsv) / np.sum(count_obsv)

    rating_dist = pd.DataFrame({"True":dist_true, "Obs.":dist_obsv},\
        columns=["True", "Obs."])
    rating_dist.plot(kind="bar", ax=ax)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    fig.savefig(os.path.join(save_root, "confounding.pdf"), bbox_inches='tight')


def user_feature_from_pref(user_pref, dimen=5, noise_std=0.1, quatize=0):
    noise = np.random.randn(*user_pref.shape)*noise_std 
    user_feat = PCA(n_components=dimen).fit_transform(user_pref+noise)
    return user_feat


data_user_dict = {
	"ml-1m" : 6000,
	"amazon-vg" : 7253,
}


def simulate():
    '''
        Basic usage:
            python simulate.py --dataset ml-1m --split 0 --plot
    '''
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="specify the dataset")
    parser.add_argument("--split", type=int, default=0,
        help="specify the split of dataset for experiment")
    parser.add_argument("--num_users" , type=int, default=-1,
        help="specify number of users in the simulated dataset")
    parser.add_argument("--confound" , type=float, default=-1,
        help="specify the strength of confounding effects")
    parser.add_argument("--plot", default=False, action="store_true",
        help="plot the exposure/rating distribution in the simulated dataset")
    parser.add_argument("--device" , type=str, default="0",
        help="specify the visible GPU device")
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

    ### Temporarily save the data in a folder
    save_root = os.path.join("temp", args.dataset)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if args.num_users == -1:
    	args.num_users = data_user_dict[args.dataset]

    ### Load rating, exposure data
    exp_data_root = os.path.join("data", args.dataset, str(args.split), "exposure")
    rat_data_root = os.path.join("data", args.dataset, str(args.split), "ratings")
    meta_table = pd.read_csv(os.path.join(exp_data_root, "meta.csv"))
    num_items = meta_table["num_items"][0]

    exp_table = pd.read_csv(os.path.join(exp_data_root, "train.csv"))
    exp_rate = get_exp_rate(exp_table)

    rat_table = pd.read_csv(os.path.join(rat_data_root, "train.csv"))
    rat_dist = get_rat_dist(rat_table)

    ### Load the pretrained exposure model
    exp_model_root = os.path.join("models", args.dataset, str(args.split), "exposure")
    exp_vae = get_collabo_vae(args.dataset, num_items)
    exp_vae.load_weights(os.path.join(exp_model_root, "best.model"))
    exp_gen = exp_vae.build_vae_gen()

    ### Load the pretrained rating model
    rat_model_root = os.path.join("models", args.dataset, str(args.split), "ratings")
    rat_vae = get_collabo_vae(args.dataset, num_items)
    rat_vae.load_weights(os.path.join(rat_model_root, "best.model"))
    rat_gen = rat_vae.build_vae_gen()

    latent_dim = rat_gen.input.shape.as_list()[-1]


    ### The confounding coefficients
    if args.confound == -1:
        conf_coeffs = np.arange(0.00, 1.10, 0.1)
    else:
        conf_coeffs = [args.confound]

    if args.dataset == "amazon-vg":
        theta_b = 2.00
    else:
        theta_b = 2.00
    print(theta_b)
    
    for conf_coeff in conf_coeffs:
        user_conf = np.random.randn(args.num_users, latent_dim)
        user_ihrt = np.random.randn(args.num_users, latent_dim)
        user_pref = conf_coeff*user_conf + (1-conf_coeff)*user_ihrt
        user_feat = user_feature_from_pref(user_pref)

        exp_raw = exp_gen.predict(user_conf)
        exp_cut = -np.sort(-exp_raw.reshape(-1))[int(args.num_users*num_items*exp_rate)]
        exp_sim = (exp_raw>exp_cut).astype(np.int32)

        rat_raw = rat_gen.predict(user_pref + theta_b*user_conf)
        rat_sim = np.zeros_like(rat_raw, dtype=np.int32)
        rat_cut_end = np.inf

        for r, rat_cut in enumerate(np.cumsum(rat_dist[::-1])):
            rat_cut_beg = -np.sort(-rat_raw.reshape(-1))[int(args.num_users*num_items*rat_cut)-1]
            rat_sim[np.logical_and(rat_raw<rat_cut_end, rat_raw>=rat_cut_beg)] = 5-r
            rat_cut_end = rat_cut_beg

        save_root = os.path.join("..", "casl_rec", "data", args.dataset, \
            str(args.split), "{:.1f}".format(conf_coeff), "raw")
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        if args.plot:
            plot_exp_stats(exp_sim, os.path.join(save_root, "plot", "exposure"), fmt="pdf")
            plot_rat_stats(rat_sim, os.path.join(save_root, "plot", "ratings"), fmt="pdf")
            plot_confounding_effects(exp_sim, rat_sim, conf_coeff, os.path.join(save_root, "plot"), fmt=".pdf")

        np.save(os.path.join(save_root, "exposure.npy"), exp_sim)
        np.save(os.path.join(save_root, "ratings.npy"), rat_sim)
        np.save(os.path.join(save_root, "raw_user_pref.npy"), user_pref)
        np.save(os.path.join(save_root, "user_features.npy"), user_feat)

        print("Done simulation {:.1f}!".format(conf_coeff))


if __name__ == '__main__':
    simulate()