import keras
from keras.models import Model

import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys
import os
import numpy as np

from mdae_models import build_model
from mdae_step import (
    build_path_and_vars,
    load_data,
)  # build_normalised_data is redefined is this function to be a vertex function

from sklearn.preprocessing import MinMaxScaler


def build_normalised_data(
    data_orig,
    data_type,
    ref_subject,
    orig_path,
    sub_list,
    index_subjects_vertices,
    train_index,
    test_index,
):
    """
    This function now runs with numpy array vectorisation (not for everything... still some index), as the for loop version took days to load
    """
    train_gyr_data = np.array([])
    train_rsfmri_data = np.array([])
    test_gyr_data = np.array([])
    test_rsfmri_data = np.array([])

    subject_gyr_data = np.array([])
    subject_rs_data = np.array([])
    print("Load testdata...")
    # seperate subjects from vertices. The arrays are supposed ordered, as any break from order wil break everything
    all_subjects = np.unique(index_subjects_vertices.T[0])
    # x = np.arrange(len(index_subjects_vertices))
    # x[train_index] = True
    # x[test_index] = False

    # a = [index_subjects_vertices.T[0] == sub for sub in all_subjects]
    # vertices_per_subject = [index_subjects_vertices.T[0][line] for line in a]
    # test_gyr_data = [
    #     load_data(
    #         data_orig,
    #         all_subjects[sub_index],
    #         4 if data_type == "gyrification" else 1,
    #         sub_list,
    #         ref_subject,
    #         orig_path,
    #     )[vertices_per_subject[sub_index]]
    #     for sub_index in len(all_subjects)
    # ]

    train_subjects_indexes = index_subjects_vertices[train_index].T[0]
    test_subjects_indexes = index_subjects_vertices[test_index].T[0]

    train_vertices = index_subjects_vertices[train_index].T[1]
    test_vertices = index_subjects_vertices[test_index].T[1]

    for subject_index in np.unique(
        index_subjects_vertices.T[
            0
        ]  # for each unique subject index, can't be rid of that for loop, need to load subject data
    ):
        # select the vertices we need for that specific subject
        subject_train_vertices = train_vertices[train_subjects_indexes == subject_index]
        subject_test_vertices = test_vertices[test_subjects_indexes == subject_index]

        # load a subject
        subject_gyr_data = load_data(
            data_orig,
            subject_index,
            4 if data_type == "gyrification" else 1,
            sub_list,
            ref_subject,
            orig_path,
        )
        subject_rs_data = load_data(
            data_orig, subject_index, 2, sub_list, ref_subject, orig_path
        )

        # Add the subject's required vertices to data
        test_gyr_data = (
            np.concatenate((test_gyr_data, subject_gyr_data[subject_test_vertices]))
            if test_gyr_data.size
            else subject_gyr_data[subject_test_vertices]
        )  # here we have to have this control because each of the modalities have a different size. time lost :/
        test_rsfmri_data = (
            np.concatenate((test_rsfmri_data, subject_rs_data[subject_test_vertices]))
            if test_rsfmri_data.size
            else subject_rs_data[subject_test_vertices]
        )
        train_gyr_data = (
            np.concatenate((train_gyr_data, subject_gyr_data[subject_train_vertices]))
            if train_gyr_data.size
            else subject_gyr_data[subject_train_vertices]
        )
        train_rsfmri_data = (
            np.concatenate((train_rsfmri_data, subject_rs_data[subject_train_vertices]))
            if train_rsfmri_data.size
            else subject_rs_data[subject_train_vertices]
        )

    scaler = MinMaxScaler()

    normalized_train_gyr_data = scaler.fit_transform(train_gyr_data)
    normalized_test_gyr_data = scaler.fit_transform(test_gyr_data)
    normalized_train_rsfmri_data = scaler.fit_transform(train_rsfmri_data)
    normalized_test_rsfmri_data = scaler.fit_transform(test_rsfmri_data)

    return (
        normalized_train_gyr_data,
        normalized_test_gyr_data,
        normalized_train_rsfmri_data,
        normalized_test_rsfmri_data,
    )


if __name__ == "__main__":

    data_orig = sys.argv[1]  # {"ABIDE", "interTVA"}
    data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    save_folder = sys.argv[3]  # Here for now the name 15-5_vertex has been chosen
    fold = int(sys.argv[4])  # int belonging to [1,10]

    (
        train_index,
        test_index,
        ref_subject,
        orig_path,
        base_path,
        index_subjects,
        sub_list,
    ) = build_path_and_vars(data_orig, data_type, "", save_folder, fold)

    # activation functions, relu / linear gives best results according to IJCNN paper, my test on dim 20 doesn't seem to change much
    hidden_layer = "relu"
    output_layer = "linear"

    index_vertices = np.arange(0, 20484)
    index_subject_vertices = np.array(
        np.meshgrid(index_subjects, index_vertices)
    ).T.reshape(-1, 2)

    print(f"Fold #{fold}")
    print(
        "Format : [subject, vertex] :\n" "TRAIN:",
        index_subject_vertices[train_index],
        "\nTEST:",
        index_subject_vertices[test_index],
    )
    # load training and testing data
    print("Load training data...")

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
        index_subject_vertices,
        train_index,
        test_index,
    )
    # Getting rid of dir here ...

    print(
        "INPUT SHAPE : ",
        normalized_train_gyr_data[0].shape,
        normalized_train_rsfmri_data[0].shape,
    )
    (
        multimodal_autoencoder,
        encoder_rsfmri,
        encoder_shared_layer,
        encoder_gyr,
    ) = build_model(
        15,  # 15 rsfMRI dimensions in latent space as determined most efficient by IJCNN paper
        5,  # 5 tfMRI dimensions in latent space as determined most efficient in IJCNN paper
        normalized_train_gyr_data[0].shape[0],  # should be 1d anyway...
        normalized_train_rsfmri_data[0].shape[0],
        hidden_layer,
        output_layer,
    )
    # fit Autoencoder on training set
    history = multimodal_autoencoder.fit(
        [normalized_train_gyr_data, normalized_train_rsfmri_data],
        [normalized_train_gyr_data, normalized_train_rsfmri_data],
        epochs=70,
        batch_size=100,
        shuffle=True,
        validation_data=(
            [normalized_test_gyr_data, normalized_test_rsfmri_data],
            [normalized_test_gyr_data, normalized_test_rsfmri_data],
        ),
    )
    # list all data in history
    print(history.history.keys())
    # save models
    # Save the results weights

    multimodal_autoencoder.save(
        "{}/{}/fold_{}/multimodal_autoencoder.h5".format(base_path, save_folder, fold)
    )
    encoder_shared_layer.save(
        "{}/{}/fold_{}/encoder_shared_layer.h5".format(base_path, save_folder, fold)
    )
    encoder_gyr.save(
        "{}/{}/fold_{}/encoder_gyr.h5".format(base_path, save_folder, fold)
    )
    encoder_rsfmri.save(
        "{}/{}/fold_{}/encoder_rsfmri.h5".format(base_path, save_folder, fold)
    )

    plt.plot(history.history["loss"], label="loss_fold_{}".format(fold))
    plt.plot(history.history["val_loss"], label="val_loss_fold_{}".format(fold))
    print("vector of val_loss", history.history["val_loss"])
    plt.title("model train vs validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("{}/{}/fold_{}/loss.png".format(base_path, save_folder, fold))
    plt.savefig("{}/{}/fold_{}/loss.pdf".format(base_path, save_folder, fold))
    plt.close()

