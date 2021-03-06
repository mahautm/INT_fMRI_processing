# !! TODO : REMOVE USE OF load_intertva_rsfmri, load_intertva_tfmri,

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
from mdae_step import (
    build_path_and_vars,
    load_raw_data,
    load_intertva_rsfmri,
    load_intertva_tfmri,
)


def build_xy_data(params, dimension, fold, sub_list):
    """
    Prepares normalised data for input and output of regression
    Parameters
    ----------
    params : dictionary
    dimension : string
        name of the dimension folder, can be a number (number of dimensions in latent space)
        two number hypenated (one for each modality), or have a word added (15-5_vertex) for
        special versions. Must match existing folder
    fold : int
        which fold folder to load data from
    sub_list : list of all subject names, in order (to correspond to Y, especially important for interTVA)

    output
    ------
    XZ : 
        normalised input matrix for regression
    YZ : 
        normalised float expected output matrix for regression (interTVA)
        or binary classification output matrix for regression (abide)

    """
    XZ = get_x_data(params, dimension, fold, subject_list=sub_list)
    Y = []
    if params["data_source"] == "ABIDE":
        classified_file = open(
            "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd_classified.json"
        )  # Hardwriten non-modifiable paths in script is bad practice. modify later !
        classified_dict = json.load(classified_file)
        # no normalisation step (which kind of seems legit for classification)
        for key in classified_dict:
            Y.append(1 if classified_dict[key] == "asd" else 0)
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


def get_x_data(
    params, dimension, fold, subject_list,
):
    """
    lazybuilder for regression input data. If data has already been calculated, it is simply loaded from file.
    Otherwise we get the model and run it once on raw data and save the data for future use before loading.

    Parameters
    ----------
    params : dictionary
    dimension : string
        name of the dimension folder, can be a number (number of dimensions in latent space)
        two number hypenated (one for each modality), or have a word added (15-5_vertex) for
        special versions. Must match existing folder
    fold : int
        which fold folder to load data from
    sub_list : list of all subject names, in order (to correspond to Y, especially important for interTVA)

    output
    ------
    X: list of floats
        input matrix for regression, loaded from the autoencoder
    """
    X = []
    input_file_path = os.path.join(
        params["orig_path"], "ae_output_{}".format(params["modality"])
    )
    for i in range(len(subject_list)):
        x_sub_data_path = os.path.join(
            input_file_path,
            str(dimension),
            "fold_{}".format(fold),
            "X_{}.npy".format(subject_list[i]),
        )
        if not os.path.exists(x_sub_data_path):
            x_sub_data = build_x_data(
                dimension, fold, subject_list, i, params, out_file=input_file_path,
            )
        else:
            x_sub_data = np.load(x_sub_data_path)
        X.append(x_sub_data)
    X = np.array(X)
    return X
    # interTVA data has already been run on taskFMRI, on frioul


def build_x_data(
    dimension, fold, subject_list, subject_index, params, out_file,
):
    """
    
    Parameters
    ----------
    dimension : string
        name of the dimension folder, can be a number (number of dimensions in latent space)
        two number hypenated (one for each modality), or have a word added (15-5_vertex) for
        special versions. Must match existing folder
    fold : int
        which fold folder to load data from
    subject : string
        subject for which data should be built as found both in subject list, corresponding to file name
    params : dictionary
    out_file : where to save the data once it is built

    output
    ------
    latent_space : resulting data latent space from the encoding layer applied on input data

    """

    # encoder_rsfmri = tf.keras.models.load_model(os.path.join(model_file_path,"/{}/fold_{}/encoder_rsfmri.h5").format(dimension,fold))
    # encoder_tfmri = tf.keras.models.load_model(os.path.join(model_file_path,"/{}/fold_{}/encoder_tfmri.h5").format(dimension,fold))
    model = tf.keras.models.load_model(
        "{}/{}/fold_{}/encoder_shared_layer.h5".format(
            params["base_path"], dimension, fold
        ),
    )

    # rsfmri data was not built with the feature extraction script, and therefore this function can fetch it from frioul

    data_1 = load_raw_data(
        params["data_source"],
        subject_index,
        # 1 is for tfMRI
        4 if params["modality"] == "gyrification" else 1,
        subject_list,
        params["ref_subject"],
        params["orig_path"],
    )
    data_2 = load_raw_data(
        params["data_source"],
        subject_index,
        # 2 is for rsfMRI
        2,
        subject_list,
        params["ref_subject"],
        params["orig_path"],
    )
    latent_space = model.predict([data_1, data_2])

    path = os.path.join(out_file, str(dimension), "fold_{}".format(fold))
    if not os.path.exists(path):
        # As we are working with parallel scripts, this will allow the script to keep working despite another one having built the directory
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    x_sub_data_path = os.path.join(path, "X_{}.npy".format(subject_list[subject_index]))
    np.save(x_sub_data_path, latent_space)
    return latent_space


def build_raw_xy_data(params, fold, sub_list):
    """
    Pulls together data which has not been through the encoder to make an input vector and an output vector which corresponds

    Parameters
    ----------
    params : dictionary
    fold : int
        which fold folder to load data from
    sub_list : string list
        list of all subject names, in order (to correspond to Y, especially important for interTVA)

    output
    ------
    X : table of floats
        input data for regression
    Y : table of floats (1,len(sublist))
        expected output for regression
    """
    # Some repeated code from load_data : not super smart
    X = get_raw_x_data(params, fold, subject_list=sub_list)
    XZ = np.array(X)
    Y = []
    if params["data_source"] == "ABIDE":
        classified_file = open(
            "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd_classified.json"
        )  # Hardwriten non-modifiable paths in script is bad practice. modify later !
        classified_dict = json.load(classified_file)
        # no normalisation step (which kind of seems legit for classification)
        for key in classified_dict:
            Y.append(1 if classified_dict[key] == "asd" else 0)
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
    x = np.array(Y)
    YZ = (x - min(x)) / (max(x) - min(x))
    return XZ, YZ


def get_raw_x_data(
    params, fold, subject_list,
):
    """
    Train everything on resting state rsfmri,
    or on gyrification, 
    or on tfmri
    or on a combination

    Parameters
    ----------
    params : dictionary
    fold : int
        which fold folder to load data from
    subject_list : string list
        list of all subject names, in order (to correspond to Y, especially important for interTVA)

    output
    ------
    X : float list
        input data for regression
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
        sub_list,
    ) = build_path_and_vars(
        params["data_source"],
        params["modality"],
        str(params["dim_1"]) + "-" + str(params["dim_2"]),
        params["fold"],
    )
    results = np.zeros(len(sub_list))
    sub_index = np.arange(0, len(sub_list))
    print("Fold #{}".format(params["fold"]))
    # Chargement des données
    X, Y = build_xy_data(params, params["dim"], params["fold"], sub_list)
    print("TRAIN:", sub_index[train_index], "TEST:", sub_index[test_index])
    # Ensemble d'entrainement
    XE = X[sub_index[train_index], :, :]
    YE = Y[sub_index[train_index]]
    # Ensemble de test
    XT = X[sub_index[test_index], :, :]
    YT = Y[sub_index[test_index]]
    beta = estimate_beta(XE, YE, params)

    file_path = "{}/regression_output/{}/{}/fold_{}/delta_{}/soft_thres_{}".format(
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
    results[sub_index[test_index]] = np.trace(
        np.transpose(XT, axes=(0, 2, 1)) @ beta, axis1=1, axis2=2
    )
    if params["data_source"] == "ABIDE":
        results[results > 0.5] = 1
        results[results <= 0.5] = 0

    # Stats
    print(results[sub_index[test_index]])
    print(
        "MSE, fold_{}".format(params["fold"]),
        mean_squared_error(YT, results[sub_index[test_index]]),
    )
    print(
        "R2 score, fold_{}".format(params["fold"]),
        r2_score(YT, results[sub_index[test_index]]),
    )
    np.save(
        os.path.join(file_path, "mse.npy"),
        mean_squared_error(YT, results[sub_index[test_index]]),
    )
    np.save(
        os.path.join(file_path, "r_squared.npy"),
        r2_score(YT, results[sub_index[test_index]]),
    )
    np.save(os.path.join(file_path, "results.npy"), results[sub_index[test_index]])

