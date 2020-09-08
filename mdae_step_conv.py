import keras
from keras.models import Model

# from keras.layers import Dropout
# from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys
import os
import numpy as np

from mdae_models import build_convolutional_model
from mdae_step import build_normalised_data, build_path_and_vars

if __name__ == "__main__":

    data_orig = sys.argv[1]  # {"ABIDE", "interTVA"}
    data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    save_folder = sys.argv[3]
    fold = int(sys.argv[4])  # int belonging to [1,10]
    dim = "15-5"
    (
        train_index,
        test_index,
        ref_subject,
        orig_path,
        base_path,
        sub_list,
    ) = build_path_and_vars(data_orig, data_type, dim, fold)

    # activation functions, relu / linear gives best results according to IJCNN paper, my test on dim 20 doesn't seem to change much
    hidden_layer = "relu"
    output_layer = "linear"
    index_subjects = np.arange(0, len(sub_list))
    print(f"Fold #{fold}")
    print(
        "TRAIN:", index_subjects[train_index], "TEST:", index_subjects[test_index],
    )
    # load training and testing data
    print("Load training data...")

    (
        normalized_train_gyr_data,
        normalized_test_gyr_data,
        normalized_train_rsfmri_data,
        normalized_test_rsfmri_data,
    ) = build_normalised_data(
        data_orig, data_type, ref_subject, orig_path, sub_list, train_index, test_index,
    )
    
    (
        multimodal_autoencoder,
        encoder_rsfmri,
        encoder_shared_layer,
        encoder_gyr,
    ) = build_convolutional_model(
        normalized_train_gyr_data[
            0
        ].shape,  # maybe here I should use len() instead of shape
        normalized_train_rsfmri_data[0].shape,
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

