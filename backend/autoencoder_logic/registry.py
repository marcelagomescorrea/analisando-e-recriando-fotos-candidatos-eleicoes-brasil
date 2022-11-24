import time
import os
import glob
import pickle
import mlflow
from tensorflow.keras import Model, models
from autoencoder_logic.params import REGISTRY_PATH, MODEL_TARGET, MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, MLFLOW_EXPERIMENT

def load_autoencoder(elected: bool, bw: bool, custom_objects: dict) -> Model:
    """
    load the latest saved autoencoder, return None if no autoencoder found
    """
    if MODEL_TARGET == "mlflow":
        stage = "Production"

        print(f"\nLoad model {stage} stage from mlflow...")

        # load model from mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        suffix = 'bw' if bw else 'color' + 'elected' if elected else 'not_elected' + 'autoencoder'

        model_uri = f"models:/{MLFLOW_MODEL_NAME + suffix}/{stage}"
        print(f"- uri: {model_uri}")

        try:
            autoencoder = mlflow.keras.load_model(model_uri=model_uri)
            print("\n✅ model loaded from mlflow")
        except:
            print(f"\n❌ no model in stage {stage} on mlflow")
            return None

        return autoencoder

    print("\nLoad autoencoder from local disk...")

   # get latest model version
    autoencoder_directory = os.path.join(REGISTRY_PATH,
        'bw' if bw else 'color',
        'elected' if elected else 'not_elected',
        'models', 'autoencoder')

    results = glob.glob(f"{autoencoder_directory}/*")
    if not results:
        return None

    autoencoder_path = sorted(results)[-1]
    print(f"- path: {autoencoder_path}")

    autoencoder = models.load_model(autoencoder_path, custom_objects=custom_objects)
    print("\n✅ autoencoder loaded from disk")

    return autoencoder

def save_autoencoder(autoencoder: Model = None,
               params: dict = None,
               metrics: dict = None,
               elected: bool = None,
               bw: bool = None) -> None:

    """
    persist trained autoencoder, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.environ.get("MODEL_TARGET") == "mlflow":

        # retrieve mlflow env params
        suffix = 'bw' if bw else 'color' + 'elected' if elected else 'not_elected' + 'autoencoder'

        # configure mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT+suffix)

        with mlflow.start_run():

            # STEP 1: push parameters to mlflow
            if params is not None:
                mlflow.log_params(params)

            # STEP 2: push metrics to mlflow
            if metrics is not None:
                mlflow.log_metrics(metrics)

            # STEP 3: push model to mlflow
            if autoencoder is not None:

                mlflow.keras.log_model(keras_model=autoencoder,
                                       artifact_path="model",
                                       keras_module="tensorflow.keras",
                                       registered_model_name=MLFLOW_MODEL_NAME+suffix)

        print("\n✅ data saved to mlflow")

        return None

    print("\nSave autoencoder to local disk...")

    # save params
    if params is not None:
        params_path = os.path.join(REGISTRY_PATH,
        'bw' if bw else 'color',
        'elected' if elected else 'not_elected',
        'params', 'autoencoder')

        if not os.path.exists(params_path):
            os.makedirs(params_path)

        print(f"- params path: {params_path}")
        with open(os.path.join(params_path,timestamp + ".pickle"), "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(REGISTRY_PATH,
        'bw' if bw else 'color',
        'elected' if elected else 'not_elected',
        'metrics', 'autoencoder')

        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)

        print(f"- metrics path: {metrics_path}")
        with open(os.path.join(metrics_path,timestamp + ".pickle"), "wb") as file:
            pickle.dump(metrics, file)

    # save autoencoder
    if autoencoder is not None:
        autoencoder_path = os.path.join(REGISTRY_PATH,
        'bw' if bw else 'color',
        'elected' if elected else 'not_elected',
        'models', 'autoencoder')

        if not os.path.exists(autoencoder_path):
            os.makedirs(autoencoder_path)

        print(f"- autoencoder path: {autoencoder_path}")
        autoencoder.save(os.path.join(autoencoder_path, timestamp + ".model"))

    print("\n✅ data saved locally")

    return None
