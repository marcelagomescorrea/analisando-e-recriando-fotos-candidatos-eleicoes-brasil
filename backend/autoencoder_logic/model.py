from typing import Tuple
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Masking
from tensorflow.keras import Model, backend as K
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from autoencoder_logic.params import AUTOENCODER_LEARNING_RATE, AUTOENCODER_N_EPOCHS, AUTOENCODER_LOSS_FACTOR, AUTOENCODER_BATCHSIZE, AUTOENCODER_PATIENCE, AUTOENCODER_VALIDATION_SPLIT

def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

def kl_loss(y_true, y_pred):
    kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis = 1)
    return kl_loss

def total_loss(y_true, y_pred):
    return AUTOENCODER_LOSS_FACTOR*r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

def build_prefix_encoder(input_shape, use_batch_norm = False, use_dropout = False):
    global K
    K.clear_session()

    '''returns an encoder model, of output_shape equals to latent_dimension'''
    encoder = Sequential()

    encoder.add(Masking(mask_value=255, input_shape=input_shape))
    encoder.add(Rescaling(1./255))

    encoder.add(Conv2D(32, (3,3), (2, 2), padding='same', activation='LeakyReLU'))
    if use_batch_norm: encoder.add(BatchNormalization())
    if use_dropout: encoder.add(Dropout(0.25))

    encoder.add(Conv2D(64, (3,3), (2, 2), padding='same', activation='LeakyReLU'))
    if use_batch_norm: encoder.add(BatchNormalization())
    if use_dropout: encoder.add(Dropout(0.25))

    encoder.add(Conv2D(64, (3,3), (2, 2), padding='same', activation='LeakyReLU'))
    if use_batch_norm: encoder.add(BatchNormalization())
    if use_dropout: encoder.add(Dropout(0.25))

    encoder.add(Conv2D(64, (3,3), (2, 2), padding='same', activation='LeakyReLU'))
    if use_batch_norm: encoder.add(BatchNormalization())
    if use_dropout: encoder.add(Dropout(0.25))

    shape_before_flattening = K.int_shape(encoder.layers[-1].output)[1:]

    encoder.add(Flatten())

    return encoder, shape_before_flattening

def build_suffix_encoder(prefix_model, latent_dimension):
  prefix_model_input = prefix_model.layers[0].input
  prefix_model_output = prefix_model.layers[-1].output

  mean_mu = Dense(latent_dimension)(prefix_model_output)
  log_var = Dense(latent_dimension)(prefix_model_output)

  # Defining a function for sampling
  def sampling(args):
    global mean_mu
    global log_var
    mean_mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.)
    return mean_mu + K.exp(log_var/2)*epsilon

  concatenate = Lambda(sampling)([mean_mu, log_var])
  return Model(prefix_model_input, concatenate)

def build_encoder(input_shape, use_batch_norm = False, use_dropout = False, latent_dimension=200):
  prefix_encoder, shape_before_flattening = build_prefix_encoder(input_shape, use_batch_norm, use_dropout)
  encoder = build_suffix_encoder(prefix_encoder, latent_dimension)
  return encoder, shape_before_flattening

def build_decoder(latent_dimension, shape_before_flattening):
  decoder = Sequential()
  decoder.add(Dense(np.prod(shape_before_flattening), input_shape=(latent_dimension, ), activation=None))
  decoder.add(Reshape(shape_before_flattening))
  decoder.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding='same', activation='LeakyReLU'))
  decoder.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding='same', activation='LeakyReLU'))
  decoder.add(Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='LeakyReLU'))
  decoder.add(Conv2DTranspose(3, kernel_size=(3,3), strides=(2,2), padding='same', activation='sigmoid'))
  return decoder

def build_autoencoder(input_shape, latent_dimension):
  encoder, shape_before_flattening = build_encoder(input_shape, True, True, latent_dimension)
  decoder = build_decoder(latent_dimension, shape_before_flattening)
  autoencoder = Model(encoder.layers[0].input, decoder(encoder.layers[-1].output))
  return autoencoder, encoder, decoder

def initialize_autoencoder(X: np.ndarray, latent_dimension: int) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    autoencoder, _, _ = build_autoencoder(X.shape[1:], latent_dimension)
    print("\n✅ autoencoder initialized")

    return autoencoder

def compile_autoencoder(autoencoder: Model) -> Model:
    """
    Compile the Neural Network
    """

    adam_optimizer = Adam(learning_rate = AUTOENCODER_LEARNING_RATE)

    autoencoder.compile(optimizer=adam_optimizer, loss = total_loss, metrics = [r_loss, kl_loss])

    print("\n✅ autoencoder compiled")
    return autoencoder

def train_autoencoder(autoencoder: Model,
                X: np.ndarray,
                y: np.ndarray) -> Tuple[Model, dict]:
    """
    Fit autoencoder and return a the tuple (fitted_autoencoder, history)
    """

    print("\nTrain autoencoder...")

    #es = EarlyStopping(patience=AUTOENCODER_PATIENCE, restore_best_weights=True, monitor='val_loss')

    history = autoencoder.fit(X,
                        y,
                        validation_split=AUTOENCODER_VALIDATION_SPLIT,
                        epochs=AUTOENCODER_N_EPOCHS,
                        batch_size=AUTOENCODER_BATCHSIZE,
                        #callbacks=[es],
                        verbose=1,
                        shuffle=True)

    print(f"\n✅ autoencoder trained ({len(X)} rows)")

    return autoencoder, history

def evaluate_autoencoder(autoencoder: Model,
                   X: np.ndarray,
                   y: np.ndarray) -> Tuple[Model, dict]:
    """
    Evaluate trained autoencoder performance on dataset
    """

    print(f"\nEvaluate autoencoder on {len(X)} rows...")

    if autoencoder is None:
        print(f"\n❌ no autoencoder to evaluate")
        return None

    metrics = autoencoder.evaluate(
        x=X,
        y=y,
        batch_size=AUTOENCODER_BATCHSIZE,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    r_loss = metrics["r_loss"]
    kl_loss = metrics["kl_loss"]

    print(f"\n✅ autoencoder evaluated: loss {round(r_loss, 2)} mae {round(kl_loss, 2)}")

    return metrics
