import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import sys
import os
import json
import errno
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.optimizers import SGD, Adadelta, Adam
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras.backend as K


def load_intertva_rsfmri(subject, path):
    # missing file creation if it is missing
    full_path = os.path.join(
        path, "correlation_matrix_fsaverage5_{}.npy".format(subject)
    )
    if not os.path.exists(full_path):
        if not os.path.exists(path):
            # As we are working with parallel scripts, this will allow the script to keep working despite another one having built the directory
            try:
                os.makedirs(path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass
        cmd = "scp mahaut.m@frioul.int.univ-amu.fr:/hpc/banco/sellami.a/InterTVA/rsfmri/{}/glm/noisefiltering/correlation_matrix_fsaverage5.npy {}".format(
            subject, full_path
        )
        os.system(cmd)
    rsfmri_data = np.load(full_path)
    return rsfmri_data


def load_intertva_tfmri(subject, path):
    # missing file creation if it is missing
    full_path = os.path.join(path, "gii_matrix_fsaverage5_{}.npy".format(subject))
    if not os.path.exists(full_path):
        if not os.path.exists(path):
            # As we are working with parallel scripts, this will allow the script to keep working despite another one having built the directory
            try:
                os.makedirs(path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

        lh_path = os.path.join(path, "gii_matrix_fsaverage5_lh_{}.npy".format(subject))
        rh_path = os.path.join(path, "gii_matrix_fsaverage5_rh_{}.npy".format(subject))
        # an exception here would maybe come in handy, in case rsync doesn't work
        cmd = "scp mahaut.m@frioul.int.univ-amu.fr:/hpc/banco/sellami.a/InterTVA/tfmri/{0}/u{0}_task-localizer_model-singletrial_denoised/gii_matrix_fsaverage5_lh.npy {1} & rsync mahaut.m@frioul.int.univ-amu.fr:/hpc/banco/sellami.a/InterTVA/tfmri/{0}/u{0}_task-localizer_model-singletrial_denoised/gii_matrix_fsaverage5_rh.npy {2}".format(
            subject, lh_path, rh_path,
        )
        os.system(cmd)
        lh_tfmri = np.load(lh_path)
        rh_tfmri = np.load(rh_path)
        # seperate frioul files can be deleted now they have been saved as a unique combined file
        cmd_deletion = "rm {} & rm {}".format(lh_path, rh_path)
        os.system(cmd_deletion)

        tfmri_data = np.concatenate((lh_tfmri, rh_tfmri))
        np.save(full_path, tfmri_data)

    else:
        tfmri_data = np.load(full_path)
    return tfmri_data


def load_data(
    data_orig,
    sub_index,
    view,
    sub_list,
    ref_sub="sub-04",
    orig_path="/scratch/mmahaut/data/intertva/",
):
    """
    The first three view are copies of Akrem's loader, but adapted to the file architecture
    found in the mesocentre, from the feature_extraction_ABIDE.py script.

    views 4 & 5 are additions that take into account the new modality, and prepare for its testing

    Parameters
    ----------
    sub:
    view: int {1,2,3,4,5}
        View 1: task fMRI
        View 2: resting-state fMRI
        View 3: concatenated views (task-fMRI + rest-fMRI)  --UNUSED
        View 4: gyrification anatomical MRI modality
        View 5: concatenated views (gyr-MRI + rest-fMRI)    --UNUSED

    ref_sub: default "USM_0050475"
        the subject the gyrification matrices were based on during the sign homogeneity phase
    
    orig_path: default "/scratch/mmahaut/data/abide/"
        where we can find the data to load


    TODO: we are dependant on global variables, sub_list and data_orig. This should be made into function or object parameters in a later version
    """
    # Import task fMRI data
    if view == 1:

        data_path = os.path.join(orig_path, "features_tfMRI")
        view_tfmri = load_intertva_tfmri(sub_list[sub_index], data_path)
        return view_tfmri

    # Import resting-state fMRI data
    elif view == 2:
        if data_orig == "ABIDE":
            view_rsfmri = np.load(
                os.path.join(
                    orig_path,
                    "features_rsfMRI/correlation_matrix_fsaverage5_{}.npy".format(
                        sub_list[sub_index]
                    ),
                )
            )

        elif data_orig == "interTVA":
            data_path = os.path.join(orig_path, "features_rsfMRI")
            view_rsfmri = load_intertva_rsfmri(sub_list[sub_index], data_path)
        return view_rsfmri

    # Import concatenated fMRI data -- UNUSED
    # elif view == 3:
    #     data_path = os.path.join(
    #         orig_path,
    #         "features_gyrification/{}_eig_vec_fsaverage5_onref_{}.npy".format(
    #             sub_list[sub_index], ref_sub
    #         ),
    #     )
    #     view_gyr = np.load(data_path)
    #     view_rsfmri = np.load(
    #         os.path.join(
    #             orig_path,
    #             "features_rsfMRI/correlation_matrix_fsaverage5_{}.npy".format(
    #                 sub_index
    #             ),
    #         )
    #     )
    #     fmri_data = np.concatenate([view_gyr, view_rsfmri], axis=1)
    #     return fmri_data

    elif view == 4:
        data_path = os.path.join(
            orig_path,
            "features_gyrification/{}_eig_vec_fsaverage5_onref_{}.npy".format(
                sub_list[sub_index], ref_sub
            ),
        )
        view_gyr = np.load(data_path)
        return view_gyr

    # -- UNUSED
    # elif view == 5:
    #     data_path = os.path.join(
    #         orig_path,
    #         "features_gyrification/{}_eig_vec_fsaverage5_onref_{}.npy".format(
    #             sub_list[sub_index], ref_sub
    #         ),
    #     )
    #     view_gyr = np.load(data_path)
    #     view_rsfmri = np.load(
    #         os.path.join(
    #             orig_path,
    #             "features_rsfMRI/correlation_matrix_fsaverage5_{}.npy".format(
    #                 sub_list[sub_index]
    #             ),
    #         )
    #     )
    #     fmri_data = np.concatenate([view_gyr, view_rsfmri], axis=1)
    #     return fmri_data


def build_normalised_data(
    data_orig,
    data_type,
    ref_subject,
    orig_path,
    sub_list,
    index_subjects,
    train_index,
    test_index,
):
    train_gyr_data = np.concatenate(
        [
            load_data(
                data_orig,
                sub_index,
                4 if data_type == "gyrification" else 1,
                sub_list,
                ref_subject,
                orig_path,
            )
            for sub_index in index_subjects[train_index]
        ]
    )
    train_rsfmri_data = np.concatenate(
        [
            load_data(data_orig, sub_index, 2, sub_list, ref_subject, orig_path)
            for sub_index in index_subjects[train_index]
        ]
    )
    print("Shape of the training data:", train_gyr_data.shape)
    print("Load testdata...")
    test_gyr_data = np.concatenate(
        [
            load_data(
                data_orig,
                sub_index,
                4 if data_type == "gyrification" else 1,
                sub_list,
                ref_subject,
                orig_path,
            )
            for sub_index in index_subjects[test_index]
        ]
    )
    test_rsfmri_data = np.concatenate(
        [
            load_data(data_orig, sub_index, 2, sub_list, ref_subject, orig_path)
            for sub_index in index_subjects[test_index]
        ]
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


def build_model(dim_1, dim_2, input_shape_1, input_shape_2, hidden_layer, output_layer):
    # Apply linear autoencoder
    # Inputs Shape
    input_view_gyr = Input(shape=(input_shape_1))
    input_view_rsfmri = Input(shape=(input_shape_2))

    # input_train_data = Input(shape=(normalized_train_data[0].shape))
    # Encoder Model
    # First view
    encoded_gyr = Dense(100, activation=hidden_layer)(input_view_gyr)  # Layer 1, View 1
    encoded_gyr = Dense(dim_1, activation=hidden_layer)(encoded_gyr)
    print("encoded gyr shape", encoded_gyr.shape)
    # Second view
    encoded_rsfmri = Dense(100, activation=hidden_layer)(
        input_view_rsfmri
    )  # Layer 1, View 2
    encoded_rsfmri = Dense(dim_2, activation=hidden_layer)(encoded_rsfmri)
    print("encoded rsfmri shape", encoded_rsfmri.shape)

    # Shared representation with concatenation !! SO here is where we change the amount of each representation
    shared_layer = concatenate(
        [encoded_gyr, encoded_rsfmri]
    )  # Layer 3: Bottelneck layer
    print("Shared Layer", shared_layer.shape)
    # output_shared_layer=Dense(dim, activation=hidden_layer)(shared_layer)
    # print("Output Shared Layer", output_shared_layer.shape)

    # Decoder Model

    decoded_gyr = Dense(dim_1, activation=hidden_layer)(shared_layer)
    decoded_gyr = Dense(100, activation=hidden_layer)(decoded_gyr)
    decoded_gyr = Dense(
        normalized_train_gyr_data[0].shape[0], activation=output_layer, name="dec_gyr",
    )(decoded_gyr)
    print("decoded_gyr", decoded_gyr.shape)
    # Second view
    decoded_rsfmri = Dense(dim_2, activation=hidden_layer)(shared_layer)
    decoded_rsfmri = Dense(100, activation=hidden_layer)(decoded_rsfmri)
    decoded_rsfmri = Dense(
        normalized_train_rsfmri_data[0].shape[0],
        activation=output_layer,
        name="dec_rsfmri",
    )(decoded_rsfmri)
    print("decoded_rsfmri", decoded_rsfmri.shape)

    # This model maps an input to its reconstruction
    multimodal_autoencoder = Model(
        inputs=[input_view_gyr, input_view_rsfmri],
        outputs=[decoded_gyr, decoded_rsfmri],
    )
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    multimodal_autoencoder.compile(optimizer=adam, loss="mse")
    print(multimodal_autoencoder.summary())
    return multimodal_autoencoder


# This is one I'm always using and that should really go in a function holder,
# might also need to be used in the mdae.py script instead of doing the writing part
def build_path_and_vars(data_orig, data_type, dim, fold):

    # Warning from previous script : That might be too many different paths. To solve that, one way would be to use os more,
    # Another would be to build a parameter object to drag everywhere, in between ? At least it is all in one place...
    if data_orig == "ABIDE":
        ref_subject = "USM_0050475"
        orig_path = "/scratch/mmahaut/data/abide/"
        base_path = "/scratch/mmahaut/data/abide/ae_gyrification"
        sub_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd.json"

    elif data_orig == "interTVA":
        ref_subject = "sub-04"
        orig_path = "/scratch/mmahaut/data/intertva/"
        ae_type = "ae" if data_type == "tfMRI" else "ae_gyrification"
        base_path = "/scratch/mmahaut/data/intertva/{}".format(ae_type)
        sub_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list.json"

    else:
        print(
            "Warning !! : Please provide data origin as parameter when calling script: either 'ABIDE' or 'interTVA' "
        )

    train_index = np.load("{}/{}/fold_{}/train_index.npy".format(base_path, dim, fold))
    test_index = np.load("{}/{}/fold_{}/test_index.npy".format(base_path, dim, fold))

    sub_list_file = open(sub_list_files)
    sub_list = json.load(sub_list_file)

    index_subjects = np.arange(0, len(sub_list))

    return (
        train_index,
        test_index,
        ref_subject,
        orig_path,
        base_path,
        index_subjects,  # This one's presence just adds a variable to return where it's not always needed, and should later be removed
        sub_list,
    )


if __name__ == "__main__":
    # Expects 3 arguements, {"ABIDE", "interTVA"}, dimension of encoding layer, fold

    data_orig = sys.argv[1]
    data_type = sys.argv[2]  # could be "tfMRI" or "gyrification"
    dim = int(sys.argv[3])
    fold = int(sys.argv[4])

    (
        train_index,
        test_index,
        ref_subject,
        orig_path,
        base_path,
        index_subjects,
        sub_list,
    ) = build_path_and_vars(data_orig, data_type, dim, fold)
    fold_path = os.path.join(base_path, str(dim), "fold_{}".format(str(fold)))

    # Should not be needed anymore, as paths are built in the mdae.py script now, waiting for an opportunity to test before removing
    # if not os.path.exists(fold_path):
    #     try:
    #         os.makedirs(fold_path)
    #     except OSError as exc:
    #         if exc.errno != errno.EEXIST:
    #             raise
    #         pass

    # activation functions, relu / linear gives best results according to IJCNN paper, my test on dim 20 doesn't seem to change much
    hidden_layer = "relu"
    output_layer = "linear"

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
        data_orig,
        data_type,
        ref_subject,
        orig_path,
        sub_list,
        index_subjects,
        train_index,
        test_index,
    )
    # Getting rid of dir here ...
    multimodal_autoencoder = build_model(
        15,
        5,
        normalized_train_gyr_data[0].shape,
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

    # This model maps an inputs to its encoded representation
    # First view
    encoder_gyr = Model(input_view_gyr, encoded_gyr)
    encoder_gyr.summary()
    # Second view
    encoder_rsfmri = Model(input_view_rsfmri, encoded_rsfmri)
    encoder_rsfmri.summary()
    # This model maps a two inputs to its bottelneck layer (shared layer)
    encoder_shared_layer = Model(
        inputs=[input_view_gyr, input_view_rsfmri], outputs=shared_layer
    )
    encoder_shared_layer.summary()
    # Separate Decoder model
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(dim,))
    # retrieve the layers of the autoencoder model
    # First view
    # decoder_gyr_layer1 = multimodal_autoencoder.layers[-6]  # Index of the first layer (after bottelneck layer)
    # decoder_gyr_layer2 = multimodal_autoencoder.layers[-4]
    # decoder_gyr_layer3 = multimodal_autoencoder.layers[-2]
    # # create the decoder model
    # decoder_gyr = Model(encoded_input, decoder_gyr_layer3(decoder_gyr_layer2(decoder_gyr_layer1(encoded_input))))
    # decoder_gyr.summary()
    # # Second view
    # decoder_rsfmri_layer1 = multimodal_autoencoder.layers[-5]
    # decoder_rsfmri_layer2 = multimodal_autoencoder.layers[-3]
    # decoder_rsfmri_layer3 = multimodal_autoencoder.layers[-1]
    # create the decoder model
    # decoder_rsfmri = Model(encoded_input, decoder_rsfmri_layer3(decoder_rsfmri_layer2(decoder_rsfmri_layer1(encoded_input))))
    # decoder_rsfmri.summary()
    multimodal_autoencoder.save(
        "{}/{}/fold_{}/multimodal_autoencoder.h5".format(base_path, dim, fold)
    )
    encoder_shared_layer.save(
        "{}/{}/fold_{}/encoder_shared_layer.h5".format(base_path, dim, fold)
    )
    encoder_gyr.save("{}/{}/fold_{}/encoder_gyr.h5".format(base_path, dim, fold))
    encoder_rsfmri.save("{}/{}/fold_{}/encoder_rsfmri.h5".format(base_path, dim, fold))
    # decoder_gyr.save('{}/fold_{}/decoder_gyr.h5'.format(dim, fold))
    # decoder_rsfmri.save('{}/fold_{}/decoder_rsfmri.h5'.format(dim, fold))
    # plot our loss
    plt.plot(history.history["loss"], label="loss_fold_{}".format(fold))
    plt.plot(history.history["val_loss"], label="val_loss_fold_{}".format(fold))
    print("vector of val_loss", history.history["val_loss"])
    plt.title("model train vs validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("{}/{}/fold_{}/loss.png".format(base_path, dim, fold))
    plt.savefig("{}/{}/fold_{}/loss.pdf".format(base_path, dim, fold))
    plt.close()

