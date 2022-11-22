import os
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.errors import NotFoundError
from tensorflow.keras.layers import Rescaling, Flatten
from pca_logic.registry import save_pca
from pca_logic.model import initialize_pca

def fit_pca(elected=True, bw=True):
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """
    print("\n⭐️ use case: fit pca")

    print("\nLoading preprocessed data...")

    # model params
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
    PCA_BATCH_SIZE = int(os.getenv("PCA_BATCH_SIZE"))
    PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS"))
    AUTOENCODER_HEIGHT = int(os.getenv("AUTOENCODER_HEIGHT"))
    AUTOENCODER_WIDTH = int(os.getenv("AUTOENCODER_WIDTH"))

    folder = os.path.join(
        os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_OUTPUT_IMG")),
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
        normalization_layer = Rescaling(1./255)
        flatten_layer = Flatten()
        normalized_images_dataset = images_dataset.map(lambda x: flatten_layer(normalization_layer(x)))
    except NotFoundError:
        print("\n✅ no data to train")
        return None

    pca = None
    #pca = load_pca(elected, bw)  # production model

    # iterate on the full dataset per chunks
    chunk_id = 0
    row_count = 0

    for image_batch in normalized_images_dataset:

        print(f"\n✅ Loading and training on preprocessed chunk n°{chunk_id}...")

        # check whether data source contain more data
        if image_batch.shape[0] < PCA_COMPONENTS:
            print(f"\nLast batch ({image_batch.shape[0]}) is no greater than pca components ({PCA_COMPONENTS}). It will be skipped.")
            break

        # increment trained row count
        chunk_row_count = image_batch.shape[0]
        row_count += chunk_row_count

        # initialize pca
        if pca is None:
            pca = initialize_pca(PCA_BATCH_SIZE, PCA_COMPONENTS)

        # train the pca incrementally
        pca = pca.partial_fit(image_batch)

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the training 👌")
        return

    print(f"\n✅ trained on {row_count} rows")

    # save pca
    save_pca(pca=pca, params=dict(PCA_BATCH_SIZE=PCA_BATCH_SIZE, PCA_COMPONENT=PCA_COMPONENTS), elected=elected, bw=bw)

    return None

def fit():
    fit_pca(True, False)
    fit_pca(False, False)
    fit_pca(True, True)
    fit_pca(False, True)
