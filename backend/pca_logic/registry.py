import os
import time
import pickle
import glob
from sklearn.decomposition import IncrementalPCA
from pca_logic.params import LOCAL_REGISTRY_PATH

def load_pca(elected: bool, bw: bool, save_copy_locally=False) -> IncrementalPCA:
    """
    load the latest saved model, return None if no model found
    """
    # if os.environ.get("PCA_TARGET") == "mlflow":
    #     stage = "Production"

    #     print(Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL)

    #     # load model from mlflow
    #     mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    #     mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

    #     model_uri = f"models:/{mlflow_model_name}/{stage}"
    #     print(f"- uri: {model_uri}")

    #     try:
    #         model = mlflow.keras.load_model(model_uri=model_uri)
    #         print("\n✅ model loaded from mlflow")
    #     except:
    #         print(f"\n❌ no model in stage {stage} on mlflow")
    #         return None

    #     if save_copy_locally:
    #         from pathlib import Path

    #         # Create the LOCAL_REGISTRY_PATH directory if it does exist
    #         Path(LOCAL_REGISTRY_PATH).mkdir(parents=True, exist_ok=True)
    #         timestamp = time.strftime("%Y%m%d-%H%M%S")
    #         model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", timestamp)
    #         model.save(model_path)

    #     return model

    print(f"\nLoad pca from local disk...")

   # get latest model version
    model_directory = os.path.join(LOCAL_REGISTRY_PATH,
        'bw' if bw else 'color',
        'elected' if elected else 'not_elected',
        'models', 'pca')

    results = glob.glob(f"{model_directory}/*")
    if not results:
        print(model_directory)
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    with open(model_path, "rb") as file:
        print("\n✅ model loaded from disk")
        return pickle.load(file)

def save_pca(pca: IncrementalPCA, params: dict, elected: bool, bw: bool) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # if os.environ.get("MODEL_TARGET") == "mlflow":

    #     # retrieve mlflow env params
    #     mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    #     mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
    #     mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

    #     # configure mlflow
    #     mlflow.set_tracking_uri(mlflow_tracking_uri)
    #     mlflow.set_experiment(experiment_name=mlflow_experiment)

    #     with mlflow.start_run():

    #         # STEP 1: push parameters to mlflow
    #         if params is not None:
    #             mlflow.log_params(params)

    #         # STEP 2: push metrics to mlflow
    #         if metrics is not None:
    #             mlflow.log_metrics(metrics)

    #         # STEP 3: push model to mlflow
    #         if model is not None:

    #             mlflow.keras.log_model(keras_model=model,
    #                                    artifact_path="model",
    #                                    keras_module="tensorflow.keras",
    #                                    registered_model_name=mlflow_model_name)

    #     print("\n✅ data saved to mlflow")

    #     return None

    print("\nSave pca to local disk...")

    # save params
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH,
        'bw' if bw else 'color',
        'elected' if elected else 'not_elected',
        'params', 'pca')
        if not os.path.exists(params_path):
            os.makedirs(params_path)
        print(f"- params path: {params_path}")
        with open(os.path.join(params_path,timestamp + ".pickle"), "wb") as file:
            pickle.dump(params, file)

    # save model
    if pca is not None:
        model_path = os.path.join(LOCAL_REGISTRY_PATH,
        'bw' if bw else 'color',
        'elected' if elected else 'not_elected',
        'models', 'pca')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print(f"- model path: {model_path}")
        with open(os.path.join(model_path,timestamp + ".pickle"), "wb") as file:
            pickle.dump(pca, file)

    print("\n✅ data saved locally")

    return None
