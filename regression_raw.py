import convexminimization as cvm
import pickle
import scipy.sparse as ssp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import tensorflow as tf
import os
import sys
import errno
import numpy as np
import json
from mdae_step import build_path_and_vars, load_intertva_rsfmri, load_intertva_tfmri

from regression import estimate_beta, load_graph, load_raw_data


######
# RUN
######

if __name__ == "__main__":

    mse = []
    rsquared = []
    # Calcul du Laplacien du graphe
    graph = load_graph()
    degree = np.array(graph.sum(axis=0))
    laplacian = ssp.diags(degree, offsets=[0]) - graph

    # Les paramètres

    params = {}

    params["data_source"] = sys.argv[1]  # ABIDE or interTVA
    params["modality"] = sys.argv[2]  # gyrification or tfMRI
    # delta is the size of Laplacian, we seek similar values in a given local brain area
    params["delta"] = float(sys.argv[3])
    # soft_tresh -- low firing neurones (below threshold) are ignored
    params["soft_thresh"] = float(sys.argv[4])
    # each fold is trained on a different node, that way calculations can be done faster
    params["fold"] = int(sys.argv[5])
    # According to IJCNN paper, 15 is the best number of dimensions for tfMRI
    params["dim_1"] = 15
    # According to IJCNN paper, 5 is the best number of dimensions for rsfMRI
    params["dim_2"] = 5
    # from feature extraction, should be fsaverage5
    params["template"] = "fsaverage5"
    params["mu_min"] = 1e-7
    params["graph"] = laplacian
    params["iterations"] = 1000
    params["mu"] = 0.1  # <- The initial descent step
    params["dim"] = "15-5"

    (
        train_index,
        test_index,  # <- train and test index are loaded to match mdae training
        params["ref_subject"],
        params["orig_path"],
        params["base_path"],
        idx,
        sub_list,
    ) = build_path_and_vars(
        params["data_source"],
        params["modality"],
        params["dim_1"],
        params["dim_2"],
        params["fold"],
    )
    results = np.zeros(len(sub_list))

    print("Fold #{}".format(params["fold"]))
    # Chargement des données
    X, Y = load_raw_data(params, params["fold"], sub_list)
    print("TRAIN:", idx[train_index], "TEST:", idx[test_index])
    # Ensemble d'entrainement
    XE = X[idx[train_index], :, :]
    YE = Y[idx[train_index]]
    # Ensemble de test
    XT = X[idx[test_index], :, :]
    YT = Y[idx[test_index]]
    beta = estimate_beta(XE, YE, params)

    file_path = "{}/regression_output/raw_input/{}/{}/fold_{}/delta_{}/soft_thres_{}".format(
        params["orig_path"],
        params["modality"],
        params["dim"],
        params["fold"],
        params["delta"],
        params["soft_thresh"],
    )
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    np.save(os.path.join(file_path, "beta.npy"), beta)

    # Estimate the results
    results[idx[test_index]] = np.trace(
        np.transpose(XT, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2
    )
    if params["data_source"] == "ABIDE":
        results[results > 0.5] = 1
        results[results <= 0.5] = 0
    # Stats
    print(results[idx[test_index]])
    print(
        "MSE, fold_{}".format(params["fold"]),
        mean_squared_error(YT, results[idx[test_index]]),
    )
    print(
        "R2 score, fold_{}".format(params["fold"]),
        r2_score(YT, results[idx[test_index]]),
    )
    np.save(
        os.path.join(file_path, "mse.npy"),
        mean_squared_error(YT, results[idx[test_index]]),
    )
    np.save(
        os.path.join(file_path, "r_squared.npy"),
        r2_score(YT, results[idx[test_index]]),
    )
    mse.append([mean_squared_error(YT, results[idx[test_index]])])
    rsquared.append([r2_score(YT, results[idx[test_index]])])

