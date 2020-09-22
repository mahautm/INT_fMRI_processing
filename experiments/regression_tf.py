import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


sys.path.append("/scratch/mmahaut/scripts/INT_fMRI_processing")
from mdae_step import build_path_and_vars
from regression import build_xy_data

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


def build_model(input_size):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=input_size),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


params = {}

params["data_source"] = "interTVA"  # ABIDE or interTVA
params["modality"] = "tfMRI"  # gyrification or tfMRI
# According to IJCNN paper, 15 is the best number of dimensions for tfMRI
params["dim_1"] = 15
# According to IJCNN paper, 5 is the best number of dimensions for rsfMRI
params["dim_2"] = 5
# from feature extraction, should be fsaverage5
params["template"] = "fsaverage5"
params["dim"] = "15-5"

y_true = [
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
y_true = np.array(y_true)
Y = (y_true - min(y_true)) / (max(y_true) - min(y_true))
y_prediction = np.zeros(39)
sub_index = np.arange(0, 39)
# Chargement des données

kf = KFold(n_splits=10)
print(kf.get_n_splits(sub_index))
print("number of splits:", kf)
orig_path = "/scratch/mmahaut/data/intertva"
fold = 0
for train_index, test_index in kf.split(sub_index):
    fold += 1
    X = []
    for i in range(3, 43):
        if i == 36:  # Avoid missing data
            continue
        mat_tf = np.load(
            "/scratch/mmahaut/data/intertva/past_data/representation_learning/relu_linear_three_layers/tfmri/10/fold_{}/X_{}.npy".format(
                fold, i
            )
        )
        mat_rsf = np.load(
            "/scratch/mmahaut/data/intertva/past_data/representation_learning/relu_linear_three_layers/rsfmri/10/fold_{}/X_{}.npy".format(
                fold, i
            )
        )
        X.append(np.concatenate((mat_tf, mat_rsf), axis=1))

    X = np.array(X)
    print("TRAIN:", sub_index[train_index], "TEST:", sub_index[test_index])
    # Ensemble d'entrainement
    XE = X[sub_index[train_index], :, :].reshape(len(train_index), -1)
    YE = Y[sub_index[train_index]]
    # Ensemble de test
    XT = X[sub_index[test_index], :, :].reshape(len(test_index), -1)
    YT = Y[sub_index[test_index]]

    # Model
    print("Train", XE.shape, "Test : ", XT.shape)
    model = build_model(XE[0].shape)
    print(model.summary())
    history = model.fit(
        XE, YE, epochs=300, verbose=0, callbacks=[tfdocs.modeling.EpochDots()],
    )
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

    # Estimate the results
    y_prediction[sub_index[test_index]] = model.predict(XT)
    print(fold, " done")
    print(y_prediction,)


# regr_tot = linear_model.LinearRegression()
# regr_tot.fit(Y.reshape(-1, 1), y_prediction)
# y_reg = regr_tot.predict(Y.reshape(-1, 1))
# # The coefficients
# print("Coefficients: \n", regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(Y, y_reg))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(Y, y_reg))
file_path = os.path.join(
    orig_path, "regression_output/tfMRI/reproducibility/tfmri10-rsfmri10/",
)

plt.scatter(Y, y_prediction, color="black")
# droite = plt.plot(Y, y_reg, color="blue")
# plt.legend(
#     [droite],
#     [
#         "Coefficients: {}\nMean squared error: %.2f \nR²: %.2f".format(regr.coef_)
#         % (mean_squared_error(Y, y_reg), r2_score(Y, y_reg))
#     ],
# )
# plt.gca().legend(loc="best")

plt.savefig(os.path.join(file_path, "y_ypred_nofista.png"))
