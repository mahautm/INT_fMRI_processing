from spektral.layers import GraphConv
from sklearn import linear_model

# from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import sys
import glob
import os
import numpy as np

# import matplotlib.pyplot as plt

sys.path.append("/scratch/mmahaut/scripts/INT_fMRI_processing/")

from regression import build_raw_xy_data, load_graph
from mdae_step import build_path_and_vars


def fold_stat(fold):
    # Calcul du Laplacien du graphe
    A = load_graph()
    params = {}

    params["data_source"] = "interTVA"  # ABIDE or interTVA
    params["modality"] = "tfMRI"  # gyrification or tfMRI
    # According to IJCNN paper, 15 is the best number of dimensions for tfMRI
    params["dim_1"] = 15
    # According to IJCNN paper, 5 is the best number of dimensions for rsfMRI
    params["dim_2"] = 5

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
        fold,
    )
    sub_index = np.arange(0, len(sub_list))

    # Load data
    X_val, y_val = build_raw_xy_data(params, fold, sub_list)
    X_test = X_val[sub_index[test_index], :, :]
    model = load_model(
        ("/scratch/mmahaut/data/intertva/ae/gcnn/fold_{}/model.h5".format(fold)),
        custom_objects={"GraphConv": GraphConv},
    )
    y_prediction = model.predict([X_test, A])
    np.save(
        "/scratch/mmahaut/data/intertva/ae/gcnn/fold_{}/predictions.npy".format(fold),
        y_prediction,
    )


if __name__ == "__main__":
    fold_stat(int(sys.argv[1]))

