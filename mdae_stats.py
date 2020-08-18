# calculates mse and rmse by comparing test data going through encoder and decoder to the expected result
# will also calculate a specific score for each modality separately, and sum them to have the global result.
# expects two parameters on run, data source and modality (see run at bottom of script)
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys
import os
import json

from sklearn.metrics import mean_squared_error

from math import sqrt
import numpy as np
import tensorflow as tf

# !! if this is not used with vertices training, use the function from mdae_step only
from mdae_step_vertices import build_normalised_data

# from mdae_step import build_normalised_data
from mdae_step import build_path_and_vars


def get_model_stats(data_orig, data_type, dimensions, number_folds="10"):
    """

    """
    # Tables:
    mse_train = []
    mse_test = []
    # RMSE (gyr+ rsfmri)
    rmse_train = []
    rmse_test = []
    #
    # Standard deviation MSE (gyr+ rsfmri)
    std_mse_train = []
    std_mse_test = []
    # Standard deviation RMSE (gyr+ rsfmri)
    std_rmse_train = []
    std_rmse_test = []
    # MSE (gyr)
    mse_gyr_train = []
    mse_gyr_test = []
    # RMSE (gyr)
    rmse_gyr_train = []
    rmse_gyr_test = []
    # std mse (gyr)
    std_mse_gyr_train = []
    std_mse_gyr_test = []
    # std rmse (gyr)
    std_rmse_gyr_train = []
    std_rmse_gyr_test = []

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

    cvscores_mse_test = []
    cvscores_rmse_test = []
    cvscores_mse_train = []
    cvscores_rmse_train = []
    cvscores_mse_gyr_train = []
    cvscores_mse_gyr_test = []
    cvscores_rmse_gyr_train = []
    cvscores_rmse_gyr_test = []
    cvscores_mse_rsfmri_train = []
    cvscores_mse_rsfmri_test = []
    cvscores_rmse_rsfmri_train = []
    cvscores_rmse_rsfmri_test = []

    # Training
    for dim in dimensions:
        for fold in range(1, number_folds + 1):

            # Vertex design changes everything, lets add some code to make it work
            (
                train_index,
                test_index,
                ref_subject,
                orig_path,
                base_path,
                index_subjects,
                sub_list,
            ) = build_path_and_vars(data_orig, data_type, "", dim, fold)
            (
                normalized_train_gyr_data,
                normalized_test_gyr_data,
                normalized_train_rsfmri_data,
                normalized_test_rsfmri_data,
            ) = build_normalised_data(
                data_orig,
                data_type,
                ref_subject,
                orig_path,
                sub_list,
                index_subjects,
                train_index,
                test_index,
            )
            # This is hard coding a temporary solution : not good

            multimodal_autoencoder = tf.keras.models.load_model(
                "{}/-{}/fold_{}/multimodal_autoencoder.h5".format(base_path, dim, fold)
            )

            print("Reconstruction of training data... ")
            [X_train_new_gyr, X_train_new_rsfmri] = multimodal_autoencoder.predict(
                [normalized_train_gyr_data, normalized_train_rsfmri_data]
            )

            # gyr
            print("Max value of predicted training gyr data ", np.max(X_train_new_gyr))
            print("Min value of predicted training gyr data", np.min(X_train_new_gyr))
            print("Reconstructed gyr matrix shape:", X_train_new_gyr.shape)
            val_mse_train_gyr = mean_squared_error(
                normalized_train_gyr_data, X_train_new_gyr
            )
            cvscores_mse_gyr_train.append(val_mse_train_gyr)
            print("Reconstruction MSE of gyr:", val_mse_train_gyr)
            val_rmse_gyr = sqrt(val_mse_train_gyr)
            print("Reconstruction RMSE of gyr : ", val_rmse_gyr)
            cvscores_rmse_gyr_train.append(val_rmse_gyr)

            # rsfmri

            print(
                "Max value of predicted training rsfmri data ",
                np.max(X_train_new_rsfmri),
            )
            print(
                "Min value of predicted training rsfmri data",
                np.min(X_train_new_rsfmri),
            )
            print("Reconstructed rsfmri matrix shape:", X_train_new_rsfmri.shape)
            val_mse_train_rsfmri = mean_squared_error(
                normalized_train_rsfmri_data, X_train_new_rsfmri
            )
            cvscores_mse_rsfmri_train.append(val_mse_train_rsfmri)
            print("Reconstruction MSE of rsfmri:", val_mse_train_rsfmri)
            val_rmse_rsfmri = sqrt(val_mse_train_rsfmri)
            print("Reconstruction RMSE of rsfmri : ", val_rmse_rsfmri)
            cvscores_rmse_rsfmri_train.append(val_rmse_rsfmri)

            # sum of MSE (gyr + rsfmri)
            cvscores_mse_train.append(np.sum([val_mse_train_gyr, val_mse_train_rsfmri]))
            # sum of RMSE (gyr + rsfmri)
            cvscores_rmse_train.append(
                sqrt(np.sum([val_mse_train_gyr, val_mse_train_rsfmri]))
            )
            # mean of MSE (gyr + rsfmri)/2

            # Reconstruction of test data
            print("Reconstruction of test data... ")
            [X_test_new_gyr, X_test_new_rsfmri] = multimodal_autoencoder.predict(
                [normalized_test_gyr_data, normalized_test_rsfmri_data]
            )

            # Test
            # gyr
            print("Max value of predicted testing gyr data ", np.max(X_test_new_gyr))
            print("Min value of predicted testing gyr data", np.min(X_test_new_gyr))
            print("Reconstructed gyr matrix shape:", X_test_new_gyr.shape)
            val_mse_test_gyr = mean_squared_error(
                normalized_test_gyr_data, X_test_new_gyr
            )
            cvscores_mse_gyr_test.append(val_mse_test_gyr)
            print("Reconstruction MSE of gyr:", val_mse_test_gyr)
            val_rmse_gyr = sqrt(val_mse_test_gyr)
            print("Reconstruction RMSE of gyr : ", val_rmse_gyr)
            cvscores_rmse_gyr_test.append(val_rmse_gyr)

            # rsfmri

            print(
                "Max value of predicted testing rsfmri data ",
                np.max(X_test_new_rsfmri),
            )
            print(
                "Min value of predicted testing rsfmri data", np.min(X_test_new_rsfmri),
            )
            print("Reconstructed rsfmri matrix shape:", X_test_new_rsfmri.shape)
            val_mse_test_rsfmri = mean_squared_error(
                normalized_test_rsfmri_data, X_test_new_rsfmri
            )
            cvscores_mse_rsfmri_test.append(val_mse_test_rsfmri)
            print("Reconstruction MSE of rsfmri:", val_mse_test_rsfmri)
            val_rmse_rsfmri = sqrt(val_mse_test_rsfmri)
            print("Reconstruction RMSE of rsfmri : ", val_rmse_rsfmri)
            cvscores_rmse_rsfmri_test.append(val_rmse_rsfmri)

            # sum of MSE (gyr + rsfmri)
            cvscores_mse_test.append(np.sum([val_mse_test_gyr, val_mse_test_rsfmri]))
            # sum of MSE (gyr + rsfmri)
            cvscores_rmse_test.append(
                sqrt(np.sum([val_mse_test_gyr, val_mse_test_rsfmri]))
            )

        # Attempt to prevent memory leak on skylake machine, legacy from when this was a loop
        # K.clear_session()

        # Save MSE, RMSE (gyr + rsfmr)
        print("shape of vector mse train", np.array([cvscores_mse_train]).shape)
        print(cvscores_mse_train)
        np.save(
            "{}/-{}/cvscores_mse_train.npy".format(base_path, dim),
            np.array([cvscores_mse_train]),
        )
        print("shape of  mse vector(test):", np.array([cvscores_mse_test]).shape)
        print(cvscores_mse_test)
        np.save(
            "{}/-{}/cvscores_mse_test.npy".format(base_path, dim),
            np.array([cvscores_mse_test]),
        )
        print("shape of rmse vector (train):", np.array([cvscores_rmse_train]).shape)
        print(cvscores_rmse_train)
        np.save(
            "{}/-{}/cvscores_rmse_train.npy".format(base_path, dim),
            np.array([cvscores_rmse_train]),
        )
        print("shape of rmse vector (test):", np.array([cvscores_rmse_test]).shape)
        print(cvscores_rmse_test)
        np.save(
            "{}/-{}/cvscores_rmse_test.npy".format(base_path, dim),
            np.array([cvscores_rmse_test]),
        )
        print(
            "%.3f%% (+/- %.5f%%)"
            % (np.mean(cvscores_mse_test), np.std(cvscores_mse_test))
        )
        mse_train.append(np.mean(cvscores_mse_train))
        std_mse_train.append(np.std(cvscores_mse_train))
        mse_test.append(np.mean(cvscores_mse_test))
        std_mse_test.append(np.std(cvscores_mse_test))
        rmse_train.append(np.mean(cvscores_rmse_train))
        std_rmse_train.append(np.std(cvscores_rmse_train))
        rmse_test.append(np.mean(cvscores_rmse_test))
        std_rmse_test.append(np.std(cvscores_rmse_test))

        # Save MSE, RMSE (gyr)
        print(
            "shape of vector mse train (gyr)", np.array([cvscores_mse_gyr_train]).shape
        )
        print(cvscores_mse_gyr_train)
        np.save(
            "{}/-{}/cvscores_mse_gyr_train.npy".format(base_path, dim),
            np.array([cvscores_mse_gyr_train]),
        )
        print("shape of  mse vector(test):", np.array([cvscores_mse_gyr_test]).shape)
        print(cvscores_mse_gyr_test)
        np.save(
            "{}/-{}/cvscores_mse_gyr_test.npy".format(base_path, dim),
            np.array([cvscores_mse_gyr_test]),
        )
        print(
            "shape of rmse vector (train):", np.array([cvscores_rmse_gyr_train]).shape
        )
        print(cvscores_rmse_gyr_train)
        np.save(
            "{}/-{}/cvscores_rmse_gyr_train.npy".format(base_path, dim),
            np.array([cvscores_rmse_gyr_test]),
        )
        print(
            "shape of rmse vector gyr (test):", np.array([cvscores_rmse_gyr_test]).shape
        )
        print(cvscores_rmse_gyr_test)
        np.save(
            "{}/-{}/cvscores_rmse_gyr_test.npy".format(base_path, dim),
            np.array([cvscores_rmse_gyr_test]),
        )
        mse_gyr_train.append(np.mean(cvscores_mse_gyr_train))
        std_mse_gyr_train.append(np.std(cvscores_mse_gyr_train))
        mse_gyr_test.append(np.mean(cvscores_mse_gyr_test))
        std_mse_gyr_test.append(np.std(cvscores_mse_gyr_test))
        rmse_gyr_train.append(np.mean(cvscores_rmse_gyr_train))
        std_rmse_gyr_train.append(np.std(cvscores_rmse_gyr_train))
        rmse_gyr_test.append(np.mean(cvscores_rmse_gyr_test))
        std_rmse_gyr_test.append(np.std(cvscores_rmse_gyr_test))

        # Save MSE, RMSE (rsfmri)
        print(
            "shape of vector mse train (rsfmri)",
            np.array([cvscores_mse_rsfmri_train]).shape,
        )
        print(cvscores_mse_rsfmri_train)
        np.save(
            "{}/-{}/cvscores_mse_rsfmri_train.npy".format(base_path, dim),
            np.array([cvscores_mse_rsfmri_train]),
        )
        print("shape of  mse vector(test):", np.array([cvscores_mse_rsfmri_test]).shape)
        print(cvscores_mse_rsfmri_test)
        np.save(
            "{}/-{}/cvscores_mse_rsfmri_test.npy".format(base_path, dim),
            np.array([cvscores_mse_rsfmri_test]),
        )
        print(
            "shape of rmse vector (train):",
            np.array([cvscores_rmse_rsfmri_train]).shape,
        )
        print(cvscores_rmse_rsfmri_train)
        np.save(
            "{}/-{}/cvscores_rmse_rsfmri_train.npy".format(base_path, dim),
            np.array([cvscores_rmse_rsfmri_test]),
        )
        print(
            "shape of rmse vector rsfmri (test):",
            np.array([cvscores_rmse_rsfmri_test]).shape,
        )
        print(cvscores_rmse_rsfmri_test)
        np.save(
            "{}/-{}/cvscores_rmse_rsfmri_test.npy".format(base_path, dim),
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
    np.save("{}/mse_train_mean.npy".format(base_path), np.array([mse_train]))
    np.save("{}/rmse_train_mean.npy".format(base_path), np.array([rmse_train]))
    np.save("{}/std_mse_train_mean.npy".format(base_path), np.array([std_mse_train]))
    np.save("{}/std_rmse_train_mean.npy".format(base_path), np.array([std_rmse_train]))
    np.save("{}/mse_test_mean.npy".format(base_path), np.array([mse_test]))
    np.save("{}/rmse_test_mean.npy".format(base_path), np.array([rmse_test]))
    np.save("{}/std_mse_test_mean.npy".format(base_path), np.array([std_mse_test]))
    np.save("{}/std_rmse_test_mean.npy".format(base_path), np.array([std_rmse_test]))

    # save MSE, RMSE, and STD vectors for training and test sets (rsfmri)

    np.save(
        "{}/mse_test_mean_rsfmri.npy".format(base_path), np.array([mse_rsfmri_test]),
    )
    np.save(
        "{}/rmse_test_mean_rsfmri.npy".format(base_path), np.array([rmse_rsfmri_test]),
    )
    np.save(
        "{}/mse_train_mean_rsfmri.npy".format(base_path), np.array([mse_rsfmri_train]),
    )
    np.save(
        "{}/rmse_train_mean_rsfmri.npy".format(base_path),
        np.array([rmse_rsfmri_train]),
    )
    np.save(
        "{}/std_mse_mean_rsfmri.npy".format(base_path), np.array([std_mse_rsfmri_test]),
    )
    np.save(
        "{}/std_rmse_mean_rsfmri.npy".format(base_path),
        np.array([std_rmse_rsfmri_test]),
    )

    # plotting the mse train
    # setting x and y axis range
    # plotting the mse train

    # Here what we really want is all the combinations between dimensions_1 and 2, and not just dimensions_1
    plt.plot(dimensions_1, mse_train, label="mse_train")
    plt.plot(dimensions_1, mse_test, label="mse_test")
    plt.xlabel("Encoding dimension")
    plt.ylabel("Reconstruction error (MSE)")
    # showing legend
    plt.legend()
    plt.savefig("{}/reconstruction_error_mse.pdf".format(base_path))
    plt.savefig("{}/reconstruction_error_mse.png".format(base_path))
    plt.close()
    # plotting the rmse train
    # setting x and y axis range
    plt.plot(dimensions_1, rmse_train, label="rmse_train")
    plt.plot(dimensions_1, rmse_test, label="rmse_test")
    plt.xlabel("Encoding dimension")
    plt.ylabel("Reconstruction error (RMSE)")
    # showing legend
    plt.legend()
    plt.savefig("{}/reconstruction_error_rmse.pdf".format(base_path))
    plt.savefig("{}/reconstruction_error_rmse.png".format(base_path))
    plt.close()


if __name__ == "__main__":
    # The dimension is used accross 3 scripts, there should be a parameter file that is loaded, probably in json format
    dimensions = ["15-5_vertex"]

    data_orig = sys.argv[1]  # Could either be "ABIDE" or "interTVA"
    data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    get_model_stats(data_orig, data_type, dimensions, 10)
