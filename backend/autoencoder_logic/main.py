import time
import numpy as np
import os
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import Model
from tensorflow.errors import NotFoundError
from autoencoder_logic.registry import save_autoencoder
from autoencoder_logic.model import initialize_autoencoder, compile_autoencoder, train_autoencoder
from autoencoder_logic.params import AUTOENCODER_BATCHSIZE,\
                                        AUTOENCODER_HEIGHT,\
                                        AUTOENCODER_WIDTH,\
                                        CHUNK_SIZE,\
                                        AUTOENCODER_PATIENCE,\
                                        AUTOENCODER_LATENT_DIMENSION,\
                                        AUTOENCODER_LEARNING_RATE, \
                                        AUTOENCODER_VALIDATION_SPLIT, \
                                        LOCAL_DATA_PATH_OUTPUT_IMG

def train():
    partial_train_autoencoder(True, False)
    partial_train_autoencoder(False, False)
    partial_train_autoencoder(True, True)
    partial_train_autoencoder(False, True)

def partial_train_autoencoder(elected=True, bw=True):
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """
    print(f"\n⭐️ use case: fit autoencoder on {'elected' if elected else 'not elected'} candidates with {'b&w' if bw else 'color'}-images")

    print("\nLoading preprocessed data...")

    folder = os.path.join(
        LOCAL_DATA_PATH_OUTPUT_IMG,
        'bw' if bw else 'color',
        'elected' if elected else 'not_elected')

    normalized_images_dataset = None
    # load a train set
    try:
        images_dataset = image_dataset_from_directory(folder,
                                                      label_mode=None,
                                                      batch_size=CHUNK_SIZE,
                                                      image_size=(AUTOENCODER_HEIGHT,AUTOENCODER_WIDTH),
                                                      shuffle=True,
                                                      crop_to_aspect_ratio=True)
    except NotFoundError:
        print("\n✅ no data to train")
        return None

    autoencoder = None
    #autoencoder = load_autoencoder(elected, bw, {'r_loss': r_loss, 'kl_loss': kl_loss, 'total_loss': total_loss})  # production model

    # iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0
    metrics_r_loss_list = []
    metrics_kl_loss_list = []

    for image_batch in images_dataset:

        print(f"\n✅ Loading and training on preprocessed chunk n°{chunk_id}...")

        # increment trained row count
        chunk_row_count = image_batch.shape[0]
        row_count += chunk_row_count

        # initialize autoencoder
        if autoencoder is None:
            autoencoder = initialize_autoencoder(image_batch, AUTOENCODER_LATENT_DIMENSION)

        # (re)compile and train the model incrementally
        autoencoder = compile_autoencoder(autoencoder)
        autoencoder, history = train_autoencoder(autoencoder,
                                     image_batch,
                                     image_batch/255)

        metrics_r_loss = np.min(history.history['r_loss'])
        metrics_kl_loss = np.min(history.history['kl_loss'])
        metrics_r_loss_list.append(metrics_r_loss)
        metrics_kl_loss_list.append(metrics_kl_loss)
        print(f"chunk r_loss, kl_loss: {round(metrics_r_loss,2)}, {round(metrics_kl_loss,2)}")

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return

    # return the last value of the validation MAE
    val_r_loss, val_kl_loss = metrics_r_loss_list[-1], metrics_kl_loss_list[-1]

    print(f"\n✅ trained on {row_count} rows with r_loss, kl_loss: {round(val_r_loss, 2)}, {round(val_kl_loss, 2)}")

    params = dict(
        # model parameters
        learning_rate=AUTOENCODER_LEARNING_RATE,
        batch_size=AUTOENCODER_BATCHSIZE,
        patience=AUTOENCODER_PATIENCE,
        # package behavior
        context="train",
        chunk_size=CHUNK_SIZE,
        # data source
        training_set_size=round(row_count*(1-AUTOENCODER_VALIDATION_SPLIT)),
        val_set_size=round(row_count*AUTOENCODER_VALIDATION_SPLIT),
        row_count=row_count,
        model_version=None,
        dataset_timestamp=time.strftime("%Y%m%d-%H%M%S"),
    )

    # save autoencoder
    save_autoencoder(autoencoder=autoencoder, params=params, metrics=dict(r_loss= val_r_loss, kl_loss=val_kl_loss), elected=elected, bw=bw)

    return val_r_loss, val_kl_loss


# def evaluate():
#     """
#     Evaluate the performance of the latest production model on new data
#     """

#     print("\n⭐️ use case: evaluate")

#     from taxifare.ml_logic.model import evaluate_model
#     from taxifare.ml_logic.registry import load_model, save_model
#     from taxifare.ml_logic.registry import get_model_version

#     # load new data
#     new_data = get_chunk(source_name=f"val_processed_{DATASET_SIZE}",
#                          index=0,
#                          chunk_size=None)  # retrieve all further data

#     if new_data is None:
#         print("\n✅ no data to evaluate")
#         return None

#     new_data = new_data.to_numpy()

#     X_new = new_data[:, :-1]
#     y_new = new_data[:, -1]

#     model = load_model()

#     metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
#     mae = metrics_dict["mae"]

#     # save evaluation
#     params = dict(
#         dataset_timestamp=get_dataset_timestamp(),
#         model_version=get_model_version(),
#         # package behavior
#         context="evaluate",
#         # data source
#         training_set_size=DATASET_SIZE,
#         val_set_size=VALIDATION_DATASET_SIZE,
#         row_count=len(X_new))

#     save_model(params=params, metrics=dict(mae=mae))

#     return mae


def pred(autoencoder: Model, X_pred: np.ndarray = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ use case: predict")

    if X_pred is None:
        decoder = Model(autoencoder.layers[-1].input, autoencoder.layers[-1].output)
        y_pred = decoder.predict(np.random.normal(0,1,size=(1,autoencoder.layers[-2].output.shape[1])))
        return y_pred

    y_pred = autoencoder.predict(X_pred)

    print("\n✅ prediction done: ", y_pred.shape)

    return y_pred
