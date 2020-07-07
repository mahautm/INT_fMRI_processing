"""""" """""" """
Monomodal Autoenconder  code
""" """""" """"""
import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys as os
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.optimizers import SGD, Adadelta, Adam
import numpy as np
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


# loading multimodal fMRI data
def load_data(sub, view):

    # Import Task fMRI data
    if view == 1:
        data_path = os.path.join(path, "tfmri/{}/gii_matrix_fsaverage5.npy".format(sub))
        if not os.path.exists(data_path):
            cmd = "rsync mahaut.m@frioul.univ-amu.fr"
            os.system(cmd)
        view_tfmri = np.load(data_path)
        return view_tfmri

    # Import Resting-State fMRI data
    if view == 2:
        view_rsfmri = np.load(
            os.path.join(
                path, "rsfmri/{}/correlation_matrix_fsaverage5.npy".format(sub)
            )
        )
        return view_rsfmri

    # Import concatenated fMRI data
    if view == 3:
        view_rsfmri = np.load(
            os.path.join(
                path, "rsfmri/{}/correlation_matrix_fsaverage5.npy".format(sub)
            )
        )
        view_tfmri = np.load(
            os.path.join(path, "tfmri/{}/gii_matrix_fsaverage5.npy".format(sub))
        )
        fmri_data = np.concatenate([view_tfmri, view_rsfmri], axis=1)
        return fmri_data


# Path
path = "d:/alpha/Cours/StageINT/fmri/"

print("View 1: task-fMRI")
print("View 2: resting-state fMRI")
print("View=3: concatenated views (task-fMRI + rest-fMRI)")


# activation functions
hidden_layer = "linear"
output_layer = "linear"

# MSE (tfmri+ rsfmri)
mse_train = []
mse_test = []
# RMSE (tfmri+ rsfmri)
rmse_train = []
rmse_test = []
#
# Standard deviation MSE (tfmri+ rsfmri)
std_mse_train = []
std_mse_test = []
# Standard deviation RMSE (tfmri+ rsfmri)
std_rmse_train = []
std_rmse_test = []


# MSE (tfmri)
mse_tfmri_train = []
mse_tfmri_test = []
# RMSE (tfmri)
rmse_tfmri_train = []
rmse_tfmri_test = []
# std mse (tfmri)
std_mse_tfmri_train = []
std_mse_tfmri_test = []
# std rmse (tfmri)
std_rmse_tfmri_train = []
std_rmse_tfmri_test = []

# MSE (rsfmri)
mse_rsfmri_train = []
mse_rsfmri_test = []
# RMSE (rsfmri)
rmse_rsfmri_train = []
rmse_rsfmri_test = []
# std mse (rsfmri)
std_mse_rsfmri_train = []
std_mse_rsfmri_test = []
# std rmse (rsfmri)
std_rmse_rsfmri_train = []
std_rmse_rsfmri_test = []

# missing data
missing_data = [36]
index_subjects = np.arange(3, 43)
index_subjects = np.delete(index_subjects, np.argwhere(index_subjects == missing_data))

# ar = np.arange(75,101)
# dimensions = ar[ar%2==0]

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
batch_2 = dimensions[6:12]
batch_3 = dimensions[12:17]
batch_4 = dimensions[17:21]
for dim in batch_1:
    # create directory
    directory = "{}".format(dim)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Cross Validation
    kf = KFold(n_splits=10)
    print(kf.get_n_splits(index_subjects))
    print("number of splits:", kf)
    print("number of features:", dimensions)
    cvscores_mse_test = []
    cvscores_rmse_test = []
    cvscores_mse_train = []
    cvscores_rmse_train = []
    cvscores_mse_tfmri_train = []
    cvscores_mse_tfmri_test = []
    cvscores_rmse_tfmri_train = []
    cvscores_rmse_tfmri_test = []
    cvscores_mse_rsfmri_train = []
    cvscores_mse_rsfmri_test = []
    cvscores_rmse_rsfmri_train = []
    cvscores_rmse_rsfmri_test = []
    fold = 0
    for train_index, test_index in kf.split(index_subjects):
        fold += 1
        # create directory
        directory = "{}/fold_{}".format(dim, fold)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(f"Fold #{fold}")
        print(
            "TRAIN:", index_subjects[train_index], "TEST:", index_subjects[test_index]
        )
        # load training and testing data
        print("Load training data...")
        train_tfmri_data = np.concatenate(
            [load_data(sub, 1) for sub in index_subjects[train_index]]
        )
        train_rsfmri_data = np.concatenate(
            [load_data(sub, 2) for sub in index_subjects[train_index]]
        )
        print("Shape of the training data:", train_tfmri_data.shape)
        print("Load testdata...")
        test_tfmri_data = np.concatenate(
            [load_data(sub, 1) for sub in index_subjects[test_index]]
        )
        test_rsfmri_data = np.concatenate(
            [load_data(sub, 2) for sub in index_subjects[test_index]]
        )
        print("Shape of the test data:", test_tfmri_data.shape)
        # Data normalization to range [-1, 1]
        print("Data normalization to range [0, 1]")
        scaler = MinMaxScaler()
        normalized_train_tfmri_data = scaler.fit_transform(train_tfmri_data)
        normalized_test_tfmri_data = scaler.fit_transform(test_tfmri_data)
        normalized_train_rsfmri_data = scaler.fit_transform(train_rsfmri_data)
        normalized_test_rsfmri_data = scaler.fit_transform(test_rsfmri_data)

        # Apply linear autoencoder
        # Inputs Shape
        input_view_tfmri = Input(shape=(normalized_train_tfmri_data[0].shape))
        input_view_rsfmri = Input(shape=(normalized_train_rsfmri_data[0].shape))

        # input_train_data = Input(shape=(normalized_train_data[0].shape))
        # Encoder Model
        # First view
        encoded_tfmri = Dense(100, activation=hidden_layer)(
            input_view_tfmri
        )  # Layer 1, View 1
        encoded_tfmri = Dense(dim, activation=hidden_layer)(encoded_tfmri)
        print("encoded tfmri shape", encoded_tfmri.shape)
        # Second view
        encoded_rsfmri = Dense(100, activation=hidden_layer)(
            input_view_rsfmri
        )  # Layer 1, View 2
        encoded_rsfmri = Dense(dim, activation=hidden_layer)(encoded_rsfmri)
        print("encoded rsfmri shape", encoded_rsfmri.shape)
        # Shared representation with concatenation
        shared_layer = concatenate(
            [encoded_tfmri, encoded_rsfmri]
        )  # Layer 3: Bottelneck layer
        print("Shared Layer", shared_layer.shape)
        # output_shared_layer=Dense(dim, activation=hidden_layer)(shared_layer)
        # print("Output Shared Layer", output_shared_layer.shape)

        # Decoder Model

        decoded_tfmri = Dense(dim, activation=hidden_layer)(shared_layer)
        decoded_tfmri = Dense(100, activation=hidden_layer)(decoded_tfmri)
        decoded_tfmri = Dense(
            normalized_train_tfmri_data[0].shape[0],
            activation=output_layer,
            name="dec_tfmri",
        )(decoded_tfmri)
        print("decoded_tfmri", decoded_tfmri.shape)
        # Second view
        decoded_rsfmri = Dense(dim, activation=hidden_layer)(shared_layer)
        decoded_rsfmri = Dense(100, activation=hidden_layer)(decoded_rsfmri)
        decoded_rsfmri = Dense(
            normalized_train_rsfmri_data[0].shape[0],
            activation=output_layer,
            name="dec_rsfmri",
        )(decoded_rsfmri)
        print("decoded_rsfmri", decoded_rsfmri.shape)

        # This model maps an input to its reconstruction
        multimodal_autoencoder = Model(
            inputs=[input_view_tfmri, input_view_rsfmri],
            outputs=[decoded_tfmri, decoded_rsfmri],
        )
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        multimodal_autoencoder.compile(optimizer=adam, loss="mse")
        print(multimodal_autoencoder.summary())
        # fit Autoencoder on training set
        history = multimodal_autoencoder.fit(
            [normalized_train_tfmri_data, normalized_train_rsfmri_data],
            [normalized_train_tfmri_data, normalized_train_rsfmri_data],
            epochs=70,
            batch_size=1000,
            shuffle=True,
            validation_data=(
                [normalized_test_tfmri_data, normalized_test_rsfmri_data],
                [normalized_test_tfmri_data, normalized_test_rsfmri_data],
            ),
        )
        # list all data in history
        print(history.history.keys())
        # save models
        # Save the results weights

        # This model maps an inputs to its encoded representation
        # First view
        encoder_tfmri = Model(input_view_tfmri, encoded_tfmri)
        encoder_tfmri.summary()
        # Second view
        encoder_rsfmri = Model(input_view_rsfmri, encoded_rsfmri)
        encoder_rsfmri.summary()
        # This model maps a two inputs to its bottelneck layer (shared layer)
        encoder_shared_layer = Model(
            inputs=[input_view_tfmri, input_view_rsfmri], outputs=shared_layer
        )
        encoder_shared_layer.summary()
        # Separate Decoder model
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(dim,))
        # retrieve the layers of the autoencoder model
        # First view
        # decoder_tfmri_layer1 = multimodal_autoencoder.layers[-6]  # Index of the first layer (after bottelneck layer)
        # decoder_tfmri_layer2 = multimodal_autoencoder.layers[-4]
        # decoder_tfmri_layer3 = multimodal_autoencoder.layers[-2]
        # # create the decoder model
        # decoder_tfmri = Model(encoded_input, decoder_tfmri_layer3(decoder_tfmri_layer2(decoder_tfmri_layer1(encoded_input))))
        # decoder_tfmri.summary()
        # # Second view
        # decoder_rsfmri_layer1 = multimodal_autoencoder.layers[-5]
        # decoder_rsfmri_layer2 = multimodal_autoencoder.layers[-3]
        # decoder_rsfmri_layer3 = multimodal_autoencoder.layers[-1]
        # create the decoder model
        # decoder_rsfmri = Model(encoded_input, decoder_rsfmri_layer3(decoder_rsfmri_layer2(decoder_rsfmri_layer1(encoded_input))))
        # decoder_rsfmri.summary()
        multimodal_autoencoder.save(
            "{}/fold_{}/multimodal_autoencoder.h5".format(dim, fold)
        )
        encoder_shared_layer.save(
            "{}/fold_{}/encoder_shared_layer.h5".format(dim, fold)
        )
        encoder_tfmri.save("{}/fold_{}/encoder_tfmri.h5".format(dim, fold))
        encoder_rsfmri.save("{}/fold_{}/encoder_rsfmri.h5".format(dim, fold))
        # decoder_tfmri.save('{}/fold_{}/decoder_tfmri.h5'.format(dim, fold))
        # decoder_rsfmri.save('{}/fold_{}/decoder_rsfmri.h5'.format(dim, fold))
        # plot our loss
        plt.plot(history.history["loss"], label="loss_fold_{}".format(fold))
        plt.plot(history.history["val_loss"], label="val_loss_fold_{}".format(fold))
        print("vector of val_loss", history.history["val_loss"])
        plt.title("model train vs validation loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig("{}/fold_{}/loss.png".format(dim, fold))
        plt.savefig("{}/fold_{}/loss.pdf".format(dim, fold))
        plt.close()

        # Reconstruction of training data
        print("Reconstruction of training data... ")
        [X_train_new_tfmri, X_train_new_rsfmri] = multimodal_autoencoder.predict(
            [normalized_train_tfmri_data, normalized_train_rsfmri_data]
        )

        # Training

        # tfmri
        print("Max value of predicted training tfmri data ", np.max(X_train_new_tfmri))
        print("Min value of predicted training tfmri data", np.min(X_train_new_tfmri))
        print("Reconstructed tfmri matrix shape:", X_train_new_tfmri.shape)
        val_mse_train_tfmri = mean_squared_error(
            normalized_train_tfmri_data, X_train_new_tfmri
        )
        cvscores_mse_tfmri_train.append(val_mse_train_tfmri)
        print("Reconstruction MSE of tfmri:", val_mse_train_tfmri)
        val_rmse_tfmri = sqrt(val_mse_train_tfmri)
        print("Reconstruction RMSE of tfmri : ", val_rmse_tfmri)
        cvscores_rmse_tfmri_train.append(val_rmse_tfmri)

        # rsfmri

        print(
            "Max value of predicted training rsfmri data ", np.max(X_train_new_rsfmri)
        )
        print("Min value of predicted training rsfmri data", np.min(X_train_new_rsfmri))
        print("Reconstructed rsfmri matrix shape:", X_train_new_rsfmri.shape)
        val_mse_train_rsfmri = mean_squared_error(
            normalized_train_rsfmri_data, X_train_new_rsfmri
        )
        cvscores_mse_rsfmri_train.append(val_mse_train_rsfmri)
        print("Reconstruction MSE of rsfmri:", val_mse_train_rsfmri)
        val_rmse_rsfmri = sqrt(val_mse_train_rsfmri)
        print("Reconstruction RMSE of rsfmri : ", val_rmse_rsfmri)
        cvscores_rmse_rsfmri_train.append(val_rmse_rsfmri)

        # sum of MSE (tfmri + rsfmri)
        cvscores_mse_train.append(np.sum([val_mse_train_tfmri, val_mse_train_rsfmri]))
        # sum of RMSE (tfmri + rsfmri)
        cvscores_rmse_train.append(
            sqrt(np.sum([val_mse_train_tfmri, val_mse_train_rsfmri]))
        )

        # Reconstruction of test data
        print("Reconstruction of test data... ")
        [X_test_new_tfmri, X_test_new_rsfmri] = multimodal_autoencoder.predict(
            [normalized_test_tfmri_data, normalized_test_rsfmri_data]
        )

        # Test
        # tfmri
        print("Max value of predicted testing tfmri data ", np.max(X_test_new_tfmri))
        print("Min value of predicted testing tfmri data", np.min(X_test_new_tfmri))
        print("Reconstructed tfmri matrix shape:", X_test_new_tfmri.shape)
        val_mse_test_tfmri = mean_squared_error(
            normalized_test_tfmri_data, X_test_new_tfmri
        )
        cvscores_mse_tfmri_test.append(val_mse_test_tfmri)
        print("Reconstruction MSE of tfmri:", val_mse_test_tfmri)
        val_rmse_tfmri = sqrt(val_mse_test_tfmri)
        print("Reconstruction RMSE of tfmri : ", val_rmse_tfmri)
        cvscores_rmse_tfmri_test.append(val_rmse_tfmri)

        # rsfmri

        print("Max value of predicted testing rsfmri data ", np.max(X_test_new_rsfmri))
        print("Min value of predicted testing rsfmri data", np.min(X_test_new_rsfmri))
        print("Reconstructed rsfmri matrix shape:", X_test_new_rsfmri.shape)
        val_mse_test_rsfmri = mean_squared_error(
            normalized_test_rsfmri_data, X_test_new_rsfmri
        )
        cvscores_mse_rsfmri_test.append(val_mse_test_rsfmri)
        print("Reconstruction MSE of rsfmri:", val_mse_test_rsfmri)
        val_rmse_rsfmri = sqrt(val_mse_test_rsfmri)
        print("Reconstruction RMSE of rsfmri : ", val_rmse_rsfmri)
        cvscores_rmse_rsfmri_test.append(val_rmse_rsfmri)

        # sum of MSE (tfmri + rsfmri)
        cvscores_mse_test.append(np.sum([val_mse_test_tfmri, val_mse_test_rsfmri]))
        # sum of MSE (tfmri + rsfmri)
        cvscores_rmse_test.append(
            sqrt(np.sum([val_mse_test_tfmri, val_mse_test_rsfmri]))
        )

    # Save MSE, RMSE (tfmri +rsfmr)
    print("shape of vector mse train", np.array([cvscores_mse_train]).shape)
    print(cvscores_mse_train)
    np.save("{}/cvscores_mse_train.npy".format(dim), np.array([cvscores_mse_train]))
    print("shape of  mse vector(test):", np.array([cvscores_mse_test]).shape)
    print(cvscores_mse_test)
    np.save("{}/cvscores_mse_test.npy".format(dim), np.array([cvscores_mse_test]))
    print("shape of rmse vector (train):", np.array([cvscores_rmse_train]).shape)
    print(cvscores_rmse_train)
    np.save("{}/cvscores_rmse_train.npy".format(dim), np.array([cvscores_rmse_train]))
    print("shape of rmse vector (test):", np.array([cvscores_rmse_test]).shape)
    print(cvscores_rmse_test)
    np.save("{}/cvscores_rmse_test.npy".format(dim), np.array([cvscores_rmse_test]))
    print(
        "%.3f%% (+/- %.5f%%)" % (np.mean(cvscores_mse_test), np.std(cvscores_mse_test))
    )
    mse_train.append(np.mean(cvscores_mse_train))
    std_mse_train.append(np.std(cvscores_mse_train))
    mse_test.append(np.mean(cvscores_mse_test))
    std_mse_test.append(np.std(cvscores_mse_test))
    rmse_train.append(np.mean(cvscores_rmse_train))
    std_rmse_train.append(np.std(cvscores_rmse_train))
    rmse_test.append(np.mean(cvscores_rmse_test))
    std_rmse_test.append(np.std(cvscores_rmse_test))

    # Save MSE, RMSE (tfmri)
    print(
        "shape of vector mse train (tfmri)", np.array([cvscores_mse_tfmri_train]).shape
    )
    print(cvscores_mse_tfmri_train)
    np.save(
        "{}/cvscores_mse_tfmri_train.npy".format(dim),
        np.array([cvscores_mse_tfmri_train]),
    )
    print("shape of  mse vector(test):", np.array([cvscores_mse_tfmri_test]).shape)
    print(cvscores_mse_tfmri_test)
    np.save(
        "{}/cvscores_mse_tfmri_test.npy".format(dim),
        np.array([cvscores_mse_tfmri_test]),
    )
    print("shape of rmse vector (train):", np.array([cvscores_rmse_tfmri_train]).shape)
    print(cvscores_rmse_tfmri_train)
    np.save(
        "{}/cvscores_rmse_tfmri_train.npy".format(dim),
        np.array([cvscores_rmse_tfmri_test]),
    )
    print(
        "shape of rmse vector tfmri (test):", np.array([cvscores_rmse_tfmri_test]).shape
    )
    print(cvscores_rmse_tfmri_test)
    np.save(
        "{}/cvscores_rmse_tfmri_test.npy".format(dim),
        np.array([cvscores_rmse_tfmri_test]),
    )
    mse_tfmri_train.append(np.mean(cvscores_mse_tfmri_train))
    std_mse_tfmri_train.append(np.std(cvscores_mse_tfmri_train))
    mse_tfmri_test.append(np.mean(cvscores_mse_tfmri_test))
    std_mse_tfmri_test.append(np.std(cvscores_mse_tfmri_test))
    rmse_tfmri_train.append(np.mean(cvscores_rmse_tfmri_train))
    std_rmse_tfmri_train.append(np.std(cvscores_rmse_tfmri_train))
    rmse_tfmri_test.append(np.mean(cvscores_rmse_tfmri_test))
    std_rmse_tfmri_test.append(np.std(cvscores_rmse_tfmri_test))

    # Save MSE, RMSE (rsfmri)
    print(
        "shape of vector mse train (rsfmri)",
        np.array([cvscores_mse_rsfmri_train]).shape,
    )
    print(cvscores_mse_rsfmri_train)
    np.save(
        "{}/cvscores_mse_rsfmri_train.npy".format(dim),
        np.array([cvscores_mse_rsfmri_train]),
    )
    print("shape of  mse vector(test):", np.array([cvscores_mse_rsfmri_test]).shape)
    print(cvscores_mse_rsfmri_test)
    np.save(
        "{}/cvscores_mse_rsfmri_test.npy".format(dim),
        np.array([cvscores_mse_rsfmri_test]),
    )
    print("shape of rmse vector (train):", np.array([cvscores_rmse_rsfmri_train]).shape)
    print(cvscores_rmse_rsfmri_train)
    np.save(
        "{}/cvscores_rmse_rsfmri_train.npy".format(dim),
        np.array([cvscores_rmse_rsfmri_test]),
    )
    print(
        "shape of rmse vector rsfmri (test):",
        np.array([cvscores_rmse_rsfmri_test]).shape,
    )
    print(cvscores_rmse_rsfmri_test)
    np.save(
        "{}/cvscores_rmse_rsfmri_test.npy".format(dim),
        np.array([cvscores_rmse_rsfmri_test]),
    )
    mse_rsfmri_train.append(np.mean(cvscores_mse_rsfmri_train))
    std_mse_rsfmri_train.append(np.std(cvscores_mse_rsfmri_train))
    mse_rsfmri_test.append(np.mean(cvscores_mse_rsfmri_test))
    std_mse_rsfmri_test.append(np.std(cvscores_mse_rsfmri_test))
    rmse_rsfmri_train.append(np.mean(cvscores_rmse_rsfmri_train))
    std_rmse_rsfmri_train.append(np.std(cvscores_rmse_rsfmri_train))
    rmse_rsfmri_test.append(np.mean(cvscores_rmse_rsfmri_test))
    std_rmse_rsfmri_test.append(np.std(cvscores_rmse_rsfmri_test))


# save MSE, RMSE, and STD vectors for training and test sets
np.save("mse_train_mean.npy", np.array([mse_train]))
np.save("rmse_train_mean.npy", np.array([rmse_train]))
np.save("std_mse_train_mean.npy", np.array([std_mse_train]))
np.save("std_rmse_train_mean.npy", np.array([std_rmse_train]))
np.save("mse_test_mean.npy", np.array([mse_test]))
np.save("rmse_test_mean.npy", np.array([rmse_test]))
np.save("std_mse_test_mean.npy", np.array([std_mse_test]))
np.save("std_rmse_test_mean.npy", np.array([std_rmse_test]))


# save MSE, RMSE, and STD vectors for training and test sets (rsfmri)

np.save("mse_test_mean_rsfmri.npy", np.array([mse_rsfmri_test]))
np.save("rmse_test_mean_rsfmri.npy", np.array([rmse_rsfmri_test]))
np.save("mse_train_mean_rsfmri.npy", np.array([mse_rsfmri_train]))
np.save("rmse_train_mean_rsfmri.npy", np.array([rmse_rsfmri_train]))
np.save("std_mse_mean_rsfmri.npy", np.array([std_mse_rsfmri_test]))
np.save("std_rmse_mean_rsfmri.npy", np.array([std_rmse_rsfmri_test]))


# plotting the mse train
# setting x and y axis range
# plotting the mse train
plt.plot(dimensions, mse_train, label="mse_train")
plt.plot(dimensions, mse_test, label="mse_test")
plt.xlabel("Encoding dimension")
plt.ylabel("Reconstruction error (MSE)")
# showing legend
plt.legend()
plt.savefig("reconstruction_error_mse.pdf")
plt.savefig("reconstruction_error_mse.png")
plt.close()
# plotting the rmse train
# setting x and y axis range
plt.plot(dimensions, rmse_train, label="rmse_train")
plt.plot(dimensions, rmse_test, label="rmse_test")
plt.xlabel("Encoding dimension")
plt.ylabel("Reconstruction error (RMSE)")
# showing legend
plt.legend()
plt.savefig("reconstruction_error_rmse.pdf")
plt.savefig("reconstruction_error_rmse.png")
plt.close()

