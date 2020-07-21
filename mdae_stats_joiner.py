import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# The dimensions are used accross 3 scripts, there should be a parameter file that is loaded, probably in json format
dimensions = [20]
data_orig = "interTVA"  # can be interTVA or ABIDE
data_type = "gyrification"  # can be gyrification or tfMRI


def joiner(parameter_list):
    # MSE (gyr+ rsfmri)
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

    # Past prints, do not work anymore
    # print(kf.get_n_splits(index_subjects))
    # print("number of splits:", kf)
    # print("number of features:", dimensions)

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

    for dim in dimensions:
        for fold in range(10):
            fold_directory = "/scratch/mmahaut/data/{}/{}/{}/{}".format(
                data_orig,
                "ae" if data_type == "tfMRI" else "ae_gyrification",
                dim,
                fold,
            )
            cvscores_mse_train = np.load(
                "{}/cvscores_mse_train.npy".format(fold_directory)
            )
            print("shape of vector mse train", cvscores_mse_train.shape)
            print(cvscores_mse_train)
            mse_train.append(np.mean(cvscores_mse_train))
            std_mse_train.append(np.std(cvscores_mse_train))

            cvscores_mse_test = np.load(
                "{}/cvscores_mse_test.npy".format(fold_directory)
            )
            print("shape of  mse vector(test):", np.array([cvscores_mse_test]).shape)
            print(cvscores_mse_test)
            mse_test.append(np.mean(cvscores_mse_test))
            std_mse_test.append(np.std(cvscores_mse_test))
            print(
                "%.3f%% (+/- %.5f%%)"
                % (np.mean(cvscores_mse_test), np.std(cvscores_mse_test))
            )

            cvscores_rmse_train = np.load(
                "{}/cvscores_rmse_train.npy".format(fold_directory)
            )
            print(
                "shape of rmse vector (train):", np.array([cvscores_rmse_train]).shape
            )
            print(cvscores_rmse_train)
            rmse_train.append(np.mean(cvscores_rmse_train))
            std_rmse_train.append(np.std(cvscores_rmse_train))

            cvscores_rmse_test = np.load(
                "{}/cvscores_rmse_test.npy".format(fold_directory)
            )
            print("shape of rmse vector (test):", np.array([cvscores_rmse_test]).shape)
            print(cvscores_rmse_test)
            rmse_test.append(np.mean(cvscores_rmse_test))
            std_rmse_test.append(np.std(cvscores_rmse_test))

    # Save MSE, RMSE (gyr)
    print("shape of vector mse train (gyr)", np.array([cvscores_mse_gyr_train]).shape)
    print(cvscores_mse_gyr_train)
    np.save(
        "{}/cvscores_mse_gyr_train.npy".format(fold_directory),
        np.array([cvscores_mse_gyr_train]),
    )
    print("shape of  mse vector(test):", np.array([cvscores_mse_gyr_test]).shape)
    print(cvscores_mse_gyr_test)
    np.save(
        "{}/cvscores_mse_gyr_test.npy".format(fold_directory),
        np.array([cvscores_mse_gyr_test]),
    )
    print("shape of rmse vector (train):", np.array([cvscores_rmse_gyr_train]).shape)
    print(cvscores_rmse_gyr_train)
    np.save(
        "{}/cvscores_rmse_gyr_train.npy".format(fold_directory),
        np.array([cvscores_rmse_gyr_test]),
    )
    print("shape of rmse vector gyr (test):", np.array([cvscores_rmse_gyr_test]).shape)
    print(cvscores_rmse_gyr_test)
    np.save(
        "{}/cvscores_rmse_gyr_test.npy".format(fold_directory),
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

    ################################# cut !
    # Save MSE, RMSE (rsfmri)
    print(
        "shape of vector mse train (rsfmri)",
        np.array([cvscores_mse_rsfmri_train]).shape,
    )
    print(cvscores_mse_rsfmri_train)
    np.save(
        "{}/cvscores_mse_rsfmri_train.npy".format(fold_directory),
        np.array([cvscores_mse_rsfmri_train]),
    )
    print("shape of  mse vector(test):", np.array([cvscores_mse_rsfmri_test]).shape)
    print(cvscores_mse_rsfmri_test)
    np.save(
        "{}/cvscores_mse_rsfmri_test.npy".format(fold_directory),
        np.array([cvscores_mse_rsfmri_test]),
    )
    print(
        "shape of rmse vector (train):", np.array([cvscores_rmse_rsfmri_train]).shape,
    )
    print(cvscores_rmse_rsfmri_train)
    np.save(
        "{}/cvscores_rmse_rsfmri_train.npy".format(fold_directory),
        np.array([cvscores_rmse_rsfmri_test]),
    )
    print(
        "shape of rmse vector rsfmri (test):",
        np.array([cvscores_rmse_rsfmri_test]).shape,
    )
    print(cvscores_rmse_rsfmri_test)
    np.save(
        "{}/cvscores_rmse_rsfmri_test.npy".format(fold_directory),
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
    ######################## From here on it is all at root, and not even at dim level
    # save MSE, RMSE, and STD vectors for training and test sets
    np.save("{}/mse_train_mean.npy".format(fold_directory), np.array([mse_train]))
    np.save("{}/rmse_train_mean.npy".format(fold_directory), np.array([rmse_train]))
    np.save(
        "{}/std_mse_train_mean.npy".format(fold_directory), np.array([std_mse_train])
    )
    np.save(
        "{}/std_rmse_train_mean.npy".format(fold_directory), np.array([std_rmse_train])
    )
    np.save("{}/mse_test_mean.npy".format(fold_directory), np.array([mse_test]))
    np.save("{}/rmse_test_mean.npy".format(fold_directory), np.array([rmse_test]))
    np.save("{}/std_mse_test_mean.npy".format(fold_directory), np.array([std_mse_test]))
    np.save(
        "{}/std_rmse_test_mean.npy".format(fold_directory), np.array([std_rmse_test])
    )

    # save MSE, RMSE, and STD vectors for training and test sets (rsfmri)

    np.save(
        "{}/mse_test_mean_rsfmri.npy".format(fold_directory),
        np.array([mse_rsfmri_test]),
    )
    np.save(
        "{}/rmse_test_mean_rsfmri.npy".format(fold_directory),
        np.array([rmse_rsfmri_test]),
    )
    np.save(
        "{}/mse_train_mean_rsfmri.npy".format(fold_directory),
        np.array([mse_rsfmri_train]),
    )
    np.save(
        "{}/rmse_train_mean_rsfmri.npy".format(fold_directory),
        np.array([rmse_rsfmri_train]),
    )
    np.save(
        "{}/std_mse_mean_rsfmri.npy".format(fold_directory),
        np.array([std_mse_rsfmri_test]),
    )
    np.save(
        "{}/std_rmse_mean_rsfmri.npy".format(fold_directory),
        np.array([std_rmse_rsfmri_test]),
    )
    ############# From past script
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
