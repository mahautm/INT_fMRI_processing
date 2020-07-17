import os

# create directory
directory = "{}/fold_{}".format(dim, fold)
if not os.path.exists(directory):
    os.makedirs(directory)
print(f"Fold #{fold}")
print(
    "TRAIN:", index_subjects[train_index], "TEST:", index_subjects[test_index],
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
encoded_gyr = Dense(100, activation=hidden_layer)(input_view_gyr)  # Layer 1, View 1
encoded_gyr = Dense(dim, activation=hidden_layer)(encoded_gyr)
print("encoded gyr shape", encoded_gyr.shape)
# Second view
encoded_rsfmri = Dense(100, activation=hidden_layer)(
    input_view_rsfmri
)  # Layer 1, View 2
encoded_rsfmri = Dense(dim, activation=hidden_layer)(encoded_rsfmri)
print("encoded rsfmri shape", encoded_rsfmri.shape)
# Shared representation with concatenation
shared_layer = concatenate([encoded_gyr, encoded_rsfmri])  # Layer 3: Bottelneck layer
print("Shared Layer", shared_layer.shape)
# output_shared_layer=Dense(dim, activation=hidden_layer)(shared_layer)
# print("Output Shared Layer", output_shared_layer.shape)

# Decoder Model

decoded_gyr = Dense(dim, activation=hidden_layer)(shared_layer)
decoded_gyr = Dense(100, activation=hidden_layer)(decoded_gyr)
decoded_gyr = Dense(
    normalized_train_gyr_data[0].shape[0], activation=output_layer, name="dec_gyr",
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
    inputs=[input_view_gyr, input_view_rsfmri], outputs=[decoded_gyr, decoded_rsfmri],
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
multimodal_autoencoder.save("{}/fold_{}/multimodal_autoencoder.h5".format(dim, fold))
encoder_shared_layer.save("{}/fold_{}/encoder_shared_layer.h5".format(dim, fold))
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
val_mse_train_gyr = mean_squared_error(normalized_train_gyr_data, X_train_new_gyr)
cvscores_mse_gyr_train.append(val_mse_train_gyr)
print("Reconstruction MSE of gyr:", val_mse_train_gyr)
val_rmse_gyr = sqrt(val_mse_train_gyr)
print("Reconstruction RMSE of gyr : ", val_rmse_gyr)
cvscores_rmse_gyr_train.append(val_rmse_gyr)

# rsfmri

print(
    "Max value of predicted training rsfmri data ", np.max(X_train_new_rsfmri),
)
print(
    "Min value of predicted training rsfmri data", np.min(X_train_new_rsfmri),
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
cvscores_rmse_train.append(sqrt(np.sum([val_mse_train_gyr, val_mse_train_rsfmri])))

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
val_mse_test_gyr = mean_squared_error(normalized_test_gyr_data, X_test_new_gyr)
cvscores_mse_gyr_test.append(val_mse_test_gyr)
print("Reconstruction MSE of gyr:", val_mse_test_gyr)
val_rmse_gyr = sqrt(val_mse_test_gyr)
print("Reconstruction RMSE of gyr : ", val_rmse_gyr)
cvscores_rmse_gyr_test.append(val_rmse_gyr)

# rsfmri

print("Max value of predicted testing rsfmri data ", np.max(X_test_new_rsfmri))
print("Min value of predicted testing rsfmri data", np.min(X_test_new_rsfmri))
print("Reconstructed rsfmri matrix shape:", X_test_new_rsfmri.shape)
val_mse_test_rsfmri = mean_squared_error(normalized_test_rsfmri_data, X_test_new_rsfmri)
cvscores_mse_rsfmri_test.append(val_mse_test_rsfmri)
print("Reconstruction MSE of rsfmri:", val_mse_test_rsfmri)
val_rmse_rsfmri = sqrt(val_mse_test_rsfmri)
print("Reconstruction RMSE of rsfmri : ", val_rmse_rsfmri)
cvscores_rmse_rsfmri_test.append(val_rmse_rsfmri)

# sum of MSE (gyr + rsfmri)
cvscores_mse_test.append(np.sum([val_mse_test_gyr, val_mse_test_rsfmri]))
# sum of MSE (gyr + rsfmri)
cvscores_rmse_test.append(sqrt(np.sum([val_mse_test_gyr, val_mse_test_rsfmri])))

# Attempt to prevent memory leak on skylake machine
K.clear_session()
