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


def load_data(params, dimension, fold, sub_file):
    """
    Prepares normalised data for input and output of regression

    """
    X = get_x_data(params, dimension, fold, subject_list=sub_file)
    # X = get_raw_x_data(params, dimension, fold, subject_list=sub_file)
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

    y = np.array(Y)
    YZ = (y - min(y)) / (max(y) - min(y))
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
            x_sub_data = np.concatenate([gyr_data, rsfmri_data])
        elif params["modality"] == "tfMRI":
            tfmri_data = load_intertva_tfmri(
                subject, os.path.join(params["orig_path"], "features_tfMRI")
            )
            x_sub_data = np.concatenate([gyr_data, tfmri_data])
        X.append(x_sub_data)

    return X
    # interTVA data has already been run on taskFMRI, on frioul


def get_x_data(
    params, dimension, fold, subject_list,
):
    """
    lazybuilder for regression input data. If data has already been calculated, it is simply loaded.
    Otherwise we get the model and run it once more
    """
    X = []
    input_file_path = os.path.join(
        params["orig_path"], "ae_output_{}".format(params["modality"])
    )

    for subject in subject_list:
        x_sub_data_path = os.path.join(
            input_file_path,
            str(dimension),
            "fold_{}".format(fold),
            "X_{}.npy".format(subject),
        )
        if not os.path.exists(x_sub_data_path):
            x_sub_data = build_x_data(
                dimension, fold, subject, params, out_file=input_file_path,
            )
        else:
            x_sub_data = np.load(x_sub_data_path)
        X.append(x_sub_data)

    return X
    # interTVA data has already been run on taskFMRI, on frioul


def build_x_data(
    dimension, fold, subject, params, out_file,
):

    # encoder_rsfmri = tf.keras.models.load_model(os.path.join(model_file_path,"/{}/fold_{}/encoder_rsfmri.h5").format(dimension,fold))
    # encoder_tfmri = tf.keras.models.load_model(os.path.join(model_file_path,"/{}/fold_{}/encoder_tfmri.h5").format(dimension,fold))
    model = tf.keras.models.load_model(
        "{}/{}/fold_{}/encoder_shared_layer.h5".format(
            params["base_path"], dimension, fold
        ),
    )

    # rsfmri data was not built with the feature extraction script, and therefore this function can fetch it from frioul
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
        prediction = model.predict([gyr_data, rsfmri_data])

    elif params["modality"] == "tfMRI":
        tfmri_data = load_intertva_tfmri(
            subject, os.path.join(params["orig_path"], "features_tfMRI")
        )
        prediction = model.predict([tfmri_data, rsfmri_data])

    path = os.path.join(out_file, str(dimension), "fold_{}".format(fold))
    if not os.path.exists(path):
        # As we are working with parallel scripts, this will allow the script to keep working despite another one having built the directory
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    x_sub_data_path = os.path.join(path, "X_{}.npy".format(subject))
    np.save(x_sub_data_path, prediction)
    return prediction


########################
# Copy of original code
########################


def load_graph():
    graph = None
    with open("/scratch/mmahaut/scripts/INT_fMRI_processing/adj_matrix.pck", "rb") as f:
        graph = pickle.load(f)
    return graph


def create_objective_function(X, Y, graph, delta):
    def objective_function(beta):
        val = 0.0
        for i in range(len(X)):
            val += 0.5 * (np.trace(X[i].T @ beta) - Y[i]) ** 2.0
        val += delta * 0.5 * np.trace(beta.T @ graph @ beta)
        return val

    return objective_function


def create_objective_function_vec(X, Y, graph, delta):
    def objective_function(beta):
        val = (
            0.5
            * (
                (np.trace(np.transpose(X, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2) - Y)
                ** 2
            ).sum()
        )
        val += delta * 0.5 * np.trace(beta.T @ graph @ beta)
        return val

    return objective_function


def create_gradient_function(X, Y, graph, delta):
    def gradient_function(beta):
        grad = beta * 0.0
        for i in range(len(X)):
            grad += X[i] * (np.trace(X[i].T @ beta) - Y[i])
        grad += delta * graph @ beta
        return grad

    return gradient_function


def create_gradient_function_vec(X, Y, graph, delta):
    def gradient_function(beta):
        grad = (
            X
            * (np.trace(np.transpose(X, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2) - Y)[
                :, np.newaxis, np.newaxis
            ]
        ).sum(axis=0)
        grad += delta * graph @ beta
        return grad

    return gradient_function


# The identity projector, i.e no constraints
def identity_projector(beta, mu):
    return beta.copy()


# The ridge projector, i.e l2 ball constraints
def create_ridge_projector(rho):
    def ridge_projector(beta, mu):
        norm_beta = np.linalg.norm(beta)
        if norm_beta <= rho:
            return beta.copy()
        return beta / norm_beta * rho


# Group sparsity projector, here sparsity on the lines
def create_group_sparsity_projector(delta):
    def group_sparsity_projector(beta, mu):
        norms = np.linalg.norm(beta, axis=1)
        idx = np.where(norms > delta)
        res = beta * 0.0
        res[idx, :] = (
            beta[idx, :]
            - np.squeeze(np.sign(beta[idx, :])) * delta * mu / norms[idx, np.newaxis]
        )
        return res

    return group_sparsity_projector


# Estimate beta from data
def estimate_beta(X, Y, params):
    objective = create_objective_function_vec(X, Y, params["graph"], params["delta"])
    gradient = create_gradient_function_vec(X, Y, params["graph"], params["delta"])
    sparse_projector = create_group_sparsity_projector(params["soft_thresh"])

    (res, mu) = cvm.monotone_fista_support(
        objective,
        gradient,
        X[4]
        * 0.0,  # Pourquoi 4 ? est-ce une valeur arbitraire, pour avoir la forme (shape) ?
        params["mu"],
        params["mu_min"],
        params["iterations"],
        sparse_projector,
    )
    return res


def cross_validation_error(X, Y, params, nbfolds=5):
    idx = np.arange(Y.shape[0])
    np.random.shuffle(idx)  # Met le bazar dans les indices
    spls = np.array_split(idx, nbfolds)  # Découpe en plusieurs morceaux
    results = np.zeros(Y.shape[0])
    for spl in spls:
        reste = np.setdiff1d(np.arange(Y.shape[0]), spl)

        # Ensemble d'entrainement
        XE = X[reste, :, :]
        YE = Y[reste]

        # Ensemble de test
        XT = X[spl, :, :]
        YT = Y[spl]

        beta = estimate_beta(XE, YE, params)

        # Estimate the results
        results[spl] = np.trace(
            np.transpose(XT, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2
        )

    return results


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
    params["mu"] = 0.1  # <- The initial descent step

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

    batch_1 = ["15-5"]

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
                params["data_source"],
                params["modality"],
                15,
                5,
                fold,  # <-- hard coded, a problem is coming here
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
            file_path = "{}/regression_output/{}/{}/fold_{}".format(
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
                    < 0.5  # <- threshold for classification
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
        file_path = "{}/regression_output/".format(params["orig_path"])
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

