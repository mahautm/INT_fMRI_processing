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

from regression import estimate_beta, load_graph


def load_data(params, dimension, fold, sub_file):
    X = get_raw_x_data(params, dimension, fold, subject_list=sub_file)
    XZ = np.array(X)

    Y = []
    if params["data_source"] == "ABIDE":
        classified_file = open(
            "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd_classified.json"
        )  # Hardwriten non-modifiable paths in script is bad practice. modify later !
        classified_dict = json.load(classified_file)
        # no normalisation step (which kind of seems legit for classification)
        for key in classified_dict:
            Y.append([1] if classified_dict[key] == "asd" else [0])
    elif params["data_source"] == "interTVA":
        # Hardcoding this array is probably not the most reusable solution...
        # Error 1 found on 30/07/2020 : bad correspondance between subject file and hardcoded Y,
        # subjects in subject file were not in the same order
        Y = [
            81.25,
            81.25,
            93.75,
            93.75,
            93.75,
            62.5,
            81.25,
            100,
            100,
            87.5,
            87.5,
            68.75,
            68.75,
            87.5,
            93.75,
            100,
            62.5,
            87.5,
            93.75,
            87.5,
            81.25,
            81.25,
            81.25,
            93.75,
            50,
            62.5,
            93.75,
            81.25,
            81.25,
            87.5,
            68.75,
            81.25,
            87.5,
            87.5,
            87.5,
            75,
            93.75,
            93.75,
            93.75,
        ]

    YZ = np.array(Y)
    return XZ, YZ


def get_raw_x_data(
    params, dimension, fold, subject_list,
):
    """
    Train everything on resting state rsfmri,
    or on gyrification, 
    or on tfmri
    or on a combination
    """
    X = []

    for subject in subject_list:
        rsfmri_data = load_intertva_rsfmri(
            subject, os.path.join(params["orig_path"], "features_rsfMRI")
        )

        if params["modality"] == "gyrification":
            gyr_data = np.load(
                os.path.join(
                    params["orig_path"],
                    "features_gyrification",
                    "{}_eig_vec_{}_onref_{}.npy".format(
                        subject, params["template"], params["ref_subject"]
                    ),
                )
            )
            x_sub_data = np.concatenate([gyr_data, rsfmri_data], axis=1)
        elif params["modality"] == "tfMRI":
            tfmri_data = load_intertva_tfmri(
                subject, os.path.join(params["orig_path"], "features_tfMRI")
            )
            x_sub_data = np.concatenate([tfmri_data, rsfmri_data], axis=1)
        X.append(x_sub_data)

    return X
    # interTVA data has already been run on taskFMRI, on frioul


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

    params["data_source"] = sys.argv[1]
    params["modality"] = sys.argv[2]
    params["template"] = "fsaverage5"

    params["mu_min"] = 1e-7
    # params["soft_thresh"] = 10e-3
    params["soft_thresh"] = 0.0
    params["delta"] = 0.0
    params["graph"] = laplacian
    params["iterations"] = 1000
    params["mu"] = 0.1

    dimensions = [
        1,
        3,
        5,
        8,
        10,
        13,
        15,
        18,
        20,
        23,
        25,
        28,
        30,
        33,
        35,
        38,
        40,
        42,
        45,
        48,
        50,
    ]

    batch_1 = [20]

    for dim in batch_1:
        # 10-fold validation

        # idx = np.arange(39)
        # kf = KFold(n_splits=10)
        # fold = 0
        results = np.zeros(39)

        # for train_index, test_index in kf.split(idx):
        # fold += 1
        for fold in range(1, 11):  # 10-fold validation
            (
                train_index,
                test_index,
                params["ref_subject"],
                params["orig_path"],
                params["base_path"],
                idx,
                sub_list,
            ) = build_path_and_vars(
                params["data_source"], params["modality"], dim, fold
            )
            print("Fold #{}".format(fold))
            # Chargement des données
            X, Y = load_data(params, dim, fold, sub_list)
            print("TRAIN:", idx[train_index], "TEST:", idx[test_index])
            # Ensemble d'entrainement
            XE = X[idx[train_index], :, :]
            YE = Y[idx[train_index]]
            # Ensemble de test
            XT = X[idx[test_index], :, :]
            YT = Y[idx[test_index]]
            beta = estimate_beta(XE, YE, params)
            file_path = "{}/regression_output/raw_input/{}/{}/fold_{}".format(
                params["orig_path"], params["modality"], dim, fold
            )
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            np.save(os.path.join(file_path, "beta.npy"), beta)

            # Estimate the results
            if params["data_source"] == "ABIDE":
                results[idx[test_index]] = (
                    0
                    if np.trace(
                        np.transpose(XT, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2
                    )
                    < 0.5
                    else 1
                )
            elif params["data_source"] == "interTVA":
                results[idx[test_index]] = np.trace(
                    np.transpose(XT, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2
                )
            print(results[idx[test_index]])
            print(
                "MSE, fold_{}".format(fold),
                mean_squared_error(YT, results[idx[test_index]]),
            )
            print(
                "R2 score, fold_{}".format(fold), r2_score(YT, results[idx[test_index]])
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

        print("mean mse {}".format(np.mean([mse])))
        file_path = "{}/regression_output/".format(params["base_path"])
        file = "mean_mse.npy"
        np.save(os.path.join(file, file_path), np.mean([mse]))
        print("mean r squared {}".format(np.mean([rsquared])))
        file = "mean_rsquared.npy"
        np.save(os.path.join(file, file_path), np.mean([rsquared]))
        print(results)
        print("Mean Error = {}".format(np.linalg.norm(results - Y) ** 0.2 / Y.shape[0]))
        print("MSE = {}".format(mean_squared_error(Y, results)))
        file = "mse.npy"
        np.save(os.path.join(file, file_path), mean_squared_error(Y, results))
        file = "r2_score.npy"
        np.save(os.path.join(file, file_path), r2_score(Y, results))

