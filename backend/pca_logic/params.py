import os

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
PCA_BATCH_SIZE = int(os.getenv("PCA_BATCH_SIZE"))
PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS"))
AUTOENCODER_HEIGHT = int(os.getenv("AUTOENCODER_HEIGHT"))
AUTOENCODER_WIDTH = int(os.getenv("AUTOENCODER_WIDTH"))
LOCAL_DATA_PATH_OUTPUT_IMG = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_OUTPUT_IMG"))
REGISTRY_PATH=os.path.expanduser(os.environ.get('LOCAL_REGISTRY_PATH')) if os.getenv('DEPLOY_TARGET') == 'local' else os.path.expanduser(os.environ.get('DOCKER_REGISTRY_PATH'))
