from sklearn.decomposition import IncrementalPCA

def initialize_pca(PCA_BATCH_SIZE:int, PCA_COMPONENTS:int) -> IncrementalPCA:
    return IncrementalPCA(batch_size=PCA_BATCH_SIZE, n_components=PCA_COMPONENTS)
