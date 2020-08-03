import keras
from keras.layers import Input, Dense, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.layers import Dropout
from keras.callbacks import EarlyStopping


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

    return multimodal_autoencoder, encoder_rsfmri, encoder_shared_layer, encoder_gyr


def build_trimodal_model():
    # Apply linear autoencoder
    # Inputs Shape
    input_view_gyr = Input(shape=(input_shape_gyr))
    input_view_rsfmri = Input(shape=(input_shape_rsfMRI))
    input_view_tfmri = Input(shape=(input_shape_tfMRI))

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
    # Third view
    encoded_tfmri = Dense(100, activation=hidden_layer)(
        input_view_tfmri
    )  # Layer 1, View 1
    encoded_tfmri = Dense(dim_3, activation=hidden_layer)(encoded_tfmri)
    print("encoded gyr shape", encoded_tfmri.shape)

    shared_layer = concatenate(
        [encoded_gyr, encoded_rsfmri, encoded_tfmri]
    )  # Layer 3: Bottelneck layer
    print("Shared Layer", shared_layer.shape)

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

    # Third view
    decoded_tfmri = Dense(dim_2, activation=hidden_layer)(shared_layer)
    decoded_tfmri = Dense(100, activation=hidden_layer)(decoded_tfmri)
    decoded_tfmri = Dense(
        normalized_train_rsfmri_data[0].shape[0],
        activation=output_layer,
        name="dec_tfmri",
    )(decoded_tfmri)
    print("decoded_tfmri", decoded_tfmri.shape)

    # This model maps an input to its reconstruction
    multimodal_autoencoder = Model(
        inputs=[input_view_gyr, input_view_rsfmri, input_view_tfmri],
        outputs=[decoded_gyr, decoded_rsfmri, decoded_tfmri],
    )
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    multimodal_autoencoder.compile(optimizer=adam, loss="mse")
    print(multimodal_autoencoder.summary())

    # This model maps an inputs to its encoded representation
    # First view
    encoder_gyr = Model(input_view_gyr, encoded_gyr)
    encoder_gyr.summary()
    # Second view
    encoder_rsfmri = Model(input_view_rsfmri, encoded_rsfmri)
    encoder_rsfmri.summary()
    # Third view
    encoder_tfmri = Model(input_view_tfmri, encoded_tfmri)
    encoder_tfmri.summary()
    # This model maps a two inputs to its bottelneck layer (shared layer)
    encoder_shared_layer = Model(
        inputs=[input_view_gyr, input_view_rsfmri, input_view_tfmri],
        outputs=shared_layer,
    )
    encoder_shared_layer.summary()

    return multimodal_autoencoder, encoder_rsfmri, encoder_shared_layer, encoder_gyr


def build_convolutional_model(
    dim_1, dim_2, input_shape_1, input_shape_2, hidden_layer, output_layer
):
    # Apply linear autoencoder
    # Inputs Shape
    input_view_1 = Input(shape=(input_shape_1))
    input_view_2 = Input(shape=(input_shape_2))

    # First view
    conv_1 = Conv2D(16, (3, 3), activation=hidden_layer, padding="same")(input_view_1)
    conv_1 = MaxPooling2D((2, 2), padding="same")(conv_1)
    conv_1 = Conv2D(8, (3, 3), activation=hidden_layer, padding="same")(conv_1)
    conv_1 = MaxPooling2D((2, 2), padding="same")(conv_1)
    conv_1 = Conv2D(8, (3, 3), activation=hidden_layer, padding="same")(conv_1)
    encoded_1 = MaxPooling2D((2, 2), padding="same")(conv_1)
    print("encoded_1 shape : ", encoded_1.shape)

    # Second view
    conv_2 = Conv2D(16, (3, 3), activation=hidden_layer, padding="same")(input_view_2)
    conv_2 = MaxPooling2D((2, 2), padding="same")(conv_2)
    conv_2 = Conv2D(8, (3, 3), activation=hidden_layer, padding="same")(conv_2)
    conv_2 = MaxPooling2D((2, 2), padding="same")(conv_2)
    conv_2 = Conv2D(8, (3, 3), activation=hidden_layer, padding="same")(conv_2)
    encoded_2 = MaxPooling2D((2, 2), padding="same")(conv_2)
    print("encoded_2 shape : ", encoded_1.shape)

    # Shared representation with concatenation
    shared_layer = concatenate([encoded_1, encoded_2])  # Layer 3: Bottelneck layer
    print("Shared Layer", shared_layer.shape)

    # Decoder Model
    # First view
    conv_1 = Conv2D(8, (3, 3), activation=hidden_layer, padding="same")(shared_layer)
    conv_1 = UpSampling2D((2, 2))(conv_1)
    conv_1 = Conv2D(8, (3, 3), activation=hidden_layer, padding="same")(conv_1)
    conv_1 = UpSampling2D((2, 2))(conv_1)
    conv_1 = Conv2D(16, (3, 3), activation=hidden_layer, padding="same")(conv_1)
    conv_1 = UpSampling2D((2, 2))(conv_1)
    decoded_1 = Conv2D(1, (3, 3), activation=output_layer, padding="same")(conv_1)

    # Second view
    conv_2 = Conv2D(8, (3, 3), activation=hidden_layer, padding="same")(shared_layer)
    conv_2 = UpSampling2D((2, 2))(conv_2)
    conv_2 = Conv2D(8, (3, 3), activation=hidden_layer, padding="same")(conv_2)
    conv_2 = UpSampling2D((2, 2))(conv_2)
    conv_2 = Conv2D(16, (3, 3), activation=hidden_layer, padding="same")(conv_2)
    conv_2 = UpSampling2D((2, 2))(conv_2)
    decoded_2 = Conv2D(1, (3, 3), activation=output_layer, padding="same")(conv_2)

    # This model maps an input to its reconstruction
    multimodal_autoencoder = Model(
        inputs=[input_view_1, input_view_2], outputs=[decoded_1, decoded_2],
    )
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    multimodal_autoencoder.compile(optimizer=adam, loss="mse")
    print(multimodal_autoencoder.summary())

    # This model maps an inputs to its encoded representation
    # First view
    encoder_1 = Model(input_view_1, encoded_1)
    encoder_1.summary()
    # Second view
    encoder_2 = Model(input_view_2, encoded_2)
    encoder_2.summary()
    # This model maps a two inputs to its bottelneck layer (shared layer)
    encoder_shared_layer = Model(
        inputs=[input_view_1, input_view_2], outputs=shared_layer
    )
    encoder_shared_layer.summary()

    return multimodal_autoencoder, encoder_1, encoder_2, encoder_shared_layer
