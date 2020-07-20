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

def load_data(
    sub_index, view, ref_sub="sub-04", orig_path="/scratch/mmahaut/data/intertva/"
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
        View 3: concatenated views (task-fMRI + rest-fMRI)    
        View 4: gyrification anatomical MRI modality
        View 5: concatenated views (gyr-MRI + rest-fMRI)

    ref_sub: default "USM_0050475"
        the subject the gyrification matrices were based on during the sign homogeneity phase
    
    orig_path: default "/scratch/mmahaut/data/abide/"
        where we can find the fata to load


    TODO: we are dependant on global variables, sub_list and data_orig. This should be made into function or object parameters in a later version
    """
    # Import anatomical gyrification data
    if view == 1:

        data_path = os.path.join(
            orig_path,
            "past_data/{}_eig_vec_fsaverage5_onref_{}.npy".format(
                sub_list[sub_index], ref_sub
            ),
        )
        view_gyr = np.load(data_path)
        return view_gyr

    # Import Resting-State fMRI data
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
            # in the past_data directory, for interTVA, subjects are solely identified by their number,
            #  we therefore remove the 'sub-' prefix at the begining of the names, as well as the 0 in single digit cases
            simplified_sub_name = (
                sub_list[sub_index][5:]
                if sub_list[sub_index][4] == "0"
                else sub_list[sub_index][4:]
            )
            view_rsfmri = np.load(
                os.path.join(
                    orig_path,
                    "past_data/rsfmri/{}/correlation_matrix_fsaverage5.npy".format(
                        simplified_sub_name
                    ),
                )
            )
        return view_rsfmri

    # Import concatenated fMRI data
    elif view == 3:
        data_path = os.path.join(
            orig_path,
            "features_gyr/{}_eig_vec_fsaverage5_onref_{}.npy".format(
                sub_list[sub_index], ref_sub
            ),
        )
        view_gyr = np.load(data_path)
        view_rsfmri = np.load(
            os.path.join(
                orig_path,
                "features_rsfMRI/correlation_matrix_fsaverage5_{}.npy".format(
                    sub_index
                ),
            )
        )
        fmri_data = np.concatenate([view_gyr, view_rsfmri], axis=1)
        return fmri_data

    elif view == 4:
        data_path = os.path.join(
            orig_path,
            "features_gyrification/{}_eig_vec_fsaverage5_onref_{}.npy".format(
                sub_list[sub_index], ref_sub
            ),
        )
        view_gyr = np.load(data_path)
        return view_gyr

    elif view == 5:
        data_path = os.path.join(
            orig_path,
            "features_gyrification/{}_eig_vec_fsaverage5_onref_{}.npy".format(
                sub_list[sub_index], ref_sub
            ),
        )
        view_gyr = np.load(data_path)
        view_rsfmri = np.load(
            os.path.join(
                orig_path,
                "features_rsfMRI/correlation_matrix_fsaverage5_{}.npy".format(
                    sub_list[sub_index]
                ),
            )
        )
        fmri_data = np.concatenate([view_gyr, view_rsfmri], axis=1)
        return fmri_data


if __name__ == "__main__":
    data_orig = sys.argv[1]
    if data_orig == "ABIDE":
        sub_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list_asd.json"
    elif data_orig == "interTVA":
        sub_list_files = "/scratch/mmahaut/scripts/INT_fMRI_processing/url_preparation/subs_list.json"
    else:
        print(
            "Warning !! : Please provide data origin as parameter when calling script: either 'ABIDE' or 'interTVA' "
        )
    sub_list_file = open(sub_list_files)
    sub_list = json.load(sub_list_file)

    # activation functions
    hidden_layer = "linear"
    output_layer = "linear"

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
            
directory = "{}".format(dim)
if not os.path.exists(directory):
os.makedirs(directory)
# Cross Validation
if data_orig == "ABIDE":
kf = StratifiedKFold(n_splits=10)
else:
kf = KFold(n_splits=10)

print(kf.get_n_splits(index_subjects))
print("number of splits:", kf)
print("number of features:", dimensions)
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
fold = 0
for train_index, test_index in kf.split(index_subjects):
fold += 1
# create directory
# os.system("sbatch /scratch/mmahaut/scripts/slurm/mdae_step.sh dim")
# create directory
directory = "{}/fold_{}".format(dim, fold)
if not os.path.exists(directory):
    os.makedirs(directory)
print(f"Fold #{fold}")
print(
    "TRAIN:",
    index_subjects[train_index],
    "TEST:",
    index_subjects[test_index],
)
# load training and testing data
print("Load training data...")

# Adding parameters for different datasets :
if data_orig == "ABIDE":
    orig_path = "/scratch/mmahaut/data/abide/"
    ref_subject = "USM_0050475"
elif data_orig == "interTVA":
    orig_path = "/scratch/mmahaut/data/intertva/"
    ref_subject = "sub-04"

train_gyr_data = np.concatenate(
    [
        load_data(sub_index, 4, ref_subject, orig_path)
        for sub_index in index_subjects[train_index]
    ]
)
train_rsfmri_data = np.concatenate(
    [
        load_data(sub_index, 2, ref_subject, orig_path)
        for sub_index in index_subjects[train_index]
    ]
)
print("Shape of the training data:", train_gyr_data.shape)
print("Load testdata...")
test_gyr_data = np.concatenate(
    [
        load_data(sub_index, 4, ref_subject, orig_path)
        for sub_index in index_subjects[test_index]
    ]
)
test_rsfmri_data = np.concatenate(
    [
        load_data(sub_index, 2, ref_subject, orig_path)
        for sub_index in index_subjects[test_index]
    ]
)
print("Shape of the test data:", test_gyr_data.shape)
# Data normalization to range [-1, 1]
print("Data normalization to range [0, 1]")
scaler = MinMaxScaler()
normalized_train_gyr_data = scaler.fit_transform(train_gyr_data)
normalized_test_gyr_data = scaler.fit_transform(test_gyr_data)
normalized_train_rsfmri_data = scaler.fit_transform(train_rsfmri_data)
normalized_test_rsfmri_data = scaler.fit_transform(test_rsfmri_data)

# Apply linear autoencoder
# Inputs Shape
input_view_gyr = Input(shape=(normalized_train_gyr_data[0].shape))
input_view_rsfmri = Input(shape=(normalized_train_rsfmri_data[0].shape))

# input_train_data = Input(shape=(normalized_train_data[0].shape))
# Encoder Model
# First view
encoded_gyr = Dense(100, activation=hidden_layer)(
    input_view_gyr
)  # Layer 1, View 1
encoded_gyr = Dense(dim, activation=hidden_layer)(encoded_gyr)
print("encoded gyr shape", encoded_gyr.shape)
# Second view
encoded_rsfmri = Dense(100, activation=hidden_layer)(
    input_view_rsfmri
)  # Layer 1, View 2
encoded_rsfmri = Dense(dim, activation=hidden_layer)(encoded_rsfmri)
print("encoded rsfmri shape", encoded_rsfmri.shape)
# Shared representation with concatenation
shared_layer = concatenate(
    [encoded_gyr, encoded_rsfmri]
)  # Layer 3: Bottelneck layer
print("Shared Layer", shared_layer.shape)
# output_shared_layer=Dense(dim, activation=hidden_layer)(shared_layer)
# print("Output Shared Layer", output_shared_layer.shape)

# Decoder Model

decoded_gyr = Dense(dim, activation=hidden_layer)(shared_layer)
decoded_gyr = Dense(100, activation=hidden_layer)(decoded_gyr)
decoded_gyr = Dense(
    normalized_train_gyr_data[0].shape[0],
    activation=output_layer,
    name="dec_gyr",
)(decoded_gyr)
print("decoded_gyr", decoded_gyr.shape)
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
    inputs=[input_view_gyr, input_view_rsfmri],
    outputs=[decoded_gyr, decoded_rsfmri],
)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
multimodal_autoencoder.compile(optimizer=adam, loss="mse")
print(multimodal_autoencoder.summary())
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
    "{}/fold_{}/multimodal_autoencoder.h5".format(dim, fold)
)
encoder_shared_layer.save(
    "{}/fold_{}/encoder_shared_layer.h5".format(dim, fold)
)
encoder_gyr.save("{}/fold_{}/encoder_gyr.h5".format(dim, fold))
encoder_rsfmri.save("{}/fold_{}/encoder_rsfmri.h5".format(dim, fold))
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
plt.savefig("{}/fold_{}/loss.png".format(dim, fold))
plt.savefig("{}/fold_{}/loss.pdf".format(dim, fold))
plt.close()

# Reconstruction of training data
print("Reconstruction of training data... ")
[X_train_new_gyr, X_train_new_rsfmri] = multimodal_autoencoder.predict(
    [normalized_train_gyr_data, normalized_train_rsfmri_data]
)

# Training

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
    "Max value of predicted testing rsfmri data ", np.max(X_test_new_rsfmri)
)
print(
    "Min value of predicted testing rsfmri data", np.min(X_test_new_rsfmri)
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

# Attempt to prevent memory leak on skylake machine
K.clear_session()

# Save MSE, RMSE (gyr +rsfmr)
print("shape of vector mse train", np.array([cvscores_mse_train]).shape)
print(cvscores_mse_train)
np.save("{}/cvscores_mse_train.npy".format(dim), np.array([cvscores_mse_train]))
print("shape of  mse vector(test):", np.array([cvscores_mse_test]).shape)
print(cvscores_mse_test)
np.save("{}/cvscores_mse_test.npy".format(dim), np.array([cvscores_mse_test]))
print("shape of rmse vector (train):", np.array([cvscores_rmse_train]).shape)
print(cvscores_rmse_train)
np.save(
"{}/cvscores_rmse_train.npy".format(dim), np.array([cvscores_rmse_train])
)
print("shape of rmse vector (test):", np.array([cvscores_rmse_test]).shape)
print(cvscores_rmse_test)
np.save("{}/cvscores_rmse_test.npy".format(dim), np.array([cvscores_rmse_test]))
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
"{}/cvscores_mse_gyr_train.npy".format(dim),
np.array([cvscores_mse_gyr_train]),
)
print("shape of  mse vector(test):", np.array([cvscores_mse_gyr_test]).shape)
print(cvscores_mse_gyr_test)
np.save(
"{}/cvscores_mse_gyr_test.npy".format(dim),
np.array([cvscores_mse_gyr_test]),
)
print(
"shape of rmse vector (train):", np.array([cvscores_rmse_gyr_train]).shape
)
print(cvscores_rmse_gyr_train)
np.save(
"{}/cvscores_rmse_gyr_train.npy".format(dim),
np.array([cvscores_rmse_gyr_test]),
)
print(
"shape of rmse vector gyr (test):", np.array([cvscores_rmse_gyr_test]).shape
)
print(cvscores_rmse_gyr_test)
np.save(
"{}/cvscores_rmse_gyr_test.npy".format(dim),
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
"{}/cvscores_mse_rsfmri_train.npy".format(dim),
np.array([cvscores_mse_rsfmri_train]),
)
print("shape of  mse vector(test):", np.array([cvscores_mse_rsfmri_test]).shape)
print(cvscores_mse_rsfmri_test)
np.save(
"{}/cvscores_mse_rsfmri_test.npy".format(dim),
np.array([cvscores_mse_rsfmri_test]),
)
print(
"shape of rmse vector (train):",
np.array([cvscores_rmse_rsfmri_train]).shape,
)
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
