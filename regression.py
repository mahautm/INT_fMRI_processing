# THIS IS ONLY FOR GYRIFICATION
import convexminimization as cvm
import pickle
import scipy.sparse as ssp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import tensorflow as tf
import os
import numpy as np
import json


def load_data(
    params,
    dimension,
    fold,
    sub_file="/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd_classified.json",
):
    # Hasn't been tested yet,
    X = get_x_data(params, dimension, fold)
    XZ = np.array(X)

    Y = []
    if params["data_source"] == "ABIDE":
        classified_file = open(sub_file)
        classified_dict = json.load(classified_file)
        # no normalisation step (which kind of seems legit for classification)
        for key, value in classified_dict:
            Y.append([1] if value == "asd" else [0])
    elif params["data_source"] == "interTVA":
        # Hardcoding this array is probably not the most reusable solution...
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


def get_x_data(
    params,
    dimension,
    fold,
    input_file_path="/scratch/mmahaut/data/intertva/ae_output_tfmri",
    subject_list_path="/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list.json",
):
    """
    lazybuilder for regression input data. If data has already been calculated, it is simply loaded.
    Otherwise we get the model and run it once more
    """
    subject_list = json.load(open(subject_list_path))
    X = []
    for subject in subject_list:
        x_sub_data_path = os.path.join(
            input_file_path,
            str(dimension),
            "fold_{}".format(fold),
            "/X_{}.npy".format(subject),
        )
        if not os.path.exists(x_sub_data_path):
            build_x_data(dimension, fold, subject, out_file=input_file_path)

        x_sub_data = np.load(x_sub_data_path)
        X.append(x_sub_data)

    return X
    # interTVA data has already been run on taskFMRI, on frioul


def build_x_data(
    dimension,
    fold,
    subject,
    model_file_path="/scratch/mmahaut/data/intertva/ae",
    rsfmri_data_file_path="/scratch/mmahaut/data/intertva/features_rsfmri",
    modality_data_file_path="/scratch/mmahaut/data/intertva/past_data/tfmri",
    out_file="",
    gyrification_suffix="_eig_vec_fsaverage5_onref_sub-04.npy",
    rsfmri_suffix="",
):

    # encoder_rsfmri = tf.keras.models.load_model(os.path.join(model_file_path,"/{}/fold_{}/encoder_rsfmri.h5").format(dimension,fold))
    # encoder_tfmri = tf.keras.models.load_model(os.path.join(model_file_path,"/{}/fold_{}/encoder_tfmri.h5").format(dimension,fold))
    model = tf.keras.models.load_model(
        "{}/{}/fold_{}/encoder_shared_layer.h5".format(
            model_file_path, dimension, fold
        ),
    )

    # rsfmri data was not built with the feature extraction script, and therefore might need to be fetched on frioul
    rsfmri_data = load_intertva_rsfmri(subject, rsfmri_data_file_path)
    if params["modality"] == "gyrification":
        gyr_data = np.load(
            os.path.join(
                modality_data_file_path, "/{}{}".format(subject, gyrification_suffix),
            )
        )
        prediction = model.predict([gyr_data, rsfmri_data])

    elif params["modality"] == "tfMRI":
        # !! This is not properly taken in the expected frioul environment, but from data given by Stéphane
        simplified_sub_name = subject[5:] if subject[4] == "0" else subject[4:]
        gyr_data = np.load(
            os.path.join(
                modality_data_file_path,
                "{}/gii_matrix_fsaverage5.npy".format(simplified_sub_name),
            )
        )
        prediction = model.predict([gyr_data, rsfmri_data])

    x_sub_data_path = os.path.join(
        out_file, str(dimension), "fold_{}".format(fold), "/X_{}.npy".format(subject),
    )
    np.save(prediction, x_sub_data_path)


def load_intertva_rsfmri(subject, path):
    # missing file creation if it is missing
    full_path = os.path.join(
        path, "correlation_matrix_fsaverage5_{}.npy".format(subject)
    )
    if not os.path.exists(full_path):
        cmd = "scp mahaut.m@frioul.int.univ-amu.fr:/hpc/banco/sellami.a/InterTVA/rsfmri/{}/glm/noisefiltering/correlation_matrix_fsaverage5.npy {}".format(
            subject, full_path
        )
        os.system(cmd)
    rsfmri_data = np.load(full_path)
    return rsfmri_data


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
        X[4] * 0.0,
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

mse = []
rsquared = []
# Calcul du Laplacien du graphe
graph = load_graph()
degree = np.array(graph.sum(axis=0))
laplacian = ssp.diags(degree, offsets=[0]) - graph

# Les paramètres
params = {}

params["data_source"] = "interTVA"
params["modality"] = "gyrification"

params["iterations"] = 1000
params["mu"] = 0.1
params["mu_min"] = 1e-7
# params["soft_thresh"] = 10e-3
params["soft_thresh"] = 0.0
params["delta"] = 0.0
params["graph"] = laplacian

sub_file_path = (
    "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd.json"
)
sub_file = open(sub_file_path)
subjects = json.load(sub_file)

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

batch_1 = dimensions[0:6]
if __name__ == "__main__":

    for dim in batch_1:
        # 10-fold validation
        idx = np.arange(len(subjects))

        # idx = np.arange(39)
        kf = KFold(n_splits=10)
        fold = 0
        results = np.zeros(39)

        for train_index, test_index in kf.split(idx):
            fold += 1
            print("Fold #{}".format(fold))
            # Chargement des données
            X, Y = load_data(params, dim, fold)
            print("TRAIN:", idx[train_index], "TEST:", idx[test_index])
            # Ensemble d'entrainement
            XE = X[idx[train_index], :, :]
            YE = Y[idx[train_index]]
            # Ensemble de test
            XT = X[idx[test_index], :, :]
            YT = Y[idx[test_index]]
            beta = estimate_beta(XE, YE, params)
            file = "fold_{}/beta.npy".format(fold)
            np.save(file, beta)
            # Estimate the results
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
            file = "fold_{}/mse.npy".format(fold)
            np.save(file, mean_squared_error(YT, results[idx[test_index]]))
            file = "fold_{}/r_squared.npy".format(fold)
            np.save(file, r2_score(YT, results[idx[test_index]]))
            mse.append([mean_squared_error(YT, results[idx[test_index]])])
            rsquared.append([r2_score(YT, results[idx[test_index]])])


print("mean mse {}".format(np.mean([mse])))
file = "mean_mse.npy"
np.save(file, np.mean([mse]))
print("mean r squared {}".format(np.mean([rsquared])))
file = "mean_rsquared.npy"
np.save(file, np.mean([rsquared]))
print(results)
print("Mean Error = {}".format(np.linalg.norm(results - Y) ** 0.2 / Y.shape[0]))
print("MSE = {}".format(mean_squared_error(Y, results)))
file = "mse.npy"
np.save(file, mean_squared_error(Y, results))
file = "r2_score.npy"
np.save(file, r2_score(Y, results))

