import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# from spektral.datasets import mnist
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor

import sys

sys.path.append("/scratch/mmahaut/scripts/INT_fMRI_processing/")

from regression import build_raw_xy_data, load_graph
from mdae_step import build_path_and_vars
import os
import numpy as np

tf.compat.v1.disable_eager_execution()
if __name__ == "__main__":

    # Calcul du Laplacien du graphe
    A = load_graph()

    # Parameters
    l2_reg = 5e-4  # Regularization rate for l2
    learning_rate = 1e-3  # Learning rate for SGD
    batch_size = 32  # Batch size
    epochs = 1000  # Number of training epochs
    es_patience = 10  # Patience fot early stopping
    params = {}

    params["data_source"] = sys.argv[1]  # ABIDE or interTVA
    params["modality"] = sys.argv[2]  # gyrification or tfMRI
    # According to IJCNN paper, 15 is the best number of dimensions for tfMRI
    params["dim_1"] = 15
    # According to IJCNN paper, 5 is the best number of dimensions for rsfMRI
    params["dim_2"] = 5

    params["fold"] = sys.argv[3]
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
    sub_index = np.arange(0, len(sub_list))

    # Load data
    X_val, y_val = build_raw_xy_data(params, params["fold"], sub_list)
    # Ensemble d'entrainement
    X_train = X_val[sub_index[train_index], :, :]
    y_train = y_val[sub_index[train_index]]
    # Ensemble de test
    X_test = X_val[sub_index[test_index], :, :]
    y_test = y_val[sub_index[test_index]]
    print(X_val.shape)
    print(X_train.shape)
    N = X_train.shape[-2]  # Number of nodes in the graphs
    print(N)
    F = X_train.shape[-1]  # Node features dimensionality
    print(F)
    n_out = 1  # Dimension of the target
    print(y_val.shape)
    fltr = GraphConv.preprocess(A)

    # Model definition
    X_in = Input(shape=(N, F))
    # Pass A as a fixed tensor, otherwise Keras will complain about inputs of
    # different rank.
    A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))

    graph_conv = GraphConv(32, activation="elu", kernel_regularizer=l2(l2_reg))(
        [X_in, A_in]
    )
    graph_conv = GraphConv(32, activation="elu", kernel_regularizer=l2(l2_reg))(
        [graph_conv, A_in]
    )
    flatten = Flatten()(graph_conv)
    fc = Dense(512, activation="relu")(flatten)
    output = Dense(n_out, activation="softmax")(fc)

    # Build model
    model = Model(inputs=[X_in, A_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["acc"])
    model.summary()

    # Train model
    validation_data = (X_val, y_val)
    model.fit(
        X_train.reshape(-1, 20484, 294),
        y_train,
        batch_size=batch_size,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=es_patience, restore_best_weights=True)],
    )

    save_path = "/scratch/mmahaut/data/intertva/ae/gcnn/fold_{}".format(params["fold"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save(os.path.join(save_path, "model.h5"))
    # Evaluate model
    print("Evaluating model.")
    eval_results = model.evaluate(X_test, y_test, batch_size=batch_size)
    print("Done.\n" "Test loss: {}\n" "Test acc: {}".format(*eval_results))
    np.save(os.path.join(save_path, "eval_results.npy"), eval_results)

