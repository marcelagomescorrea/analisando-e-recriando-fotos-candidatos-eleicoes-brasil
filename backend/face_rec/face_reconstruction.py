import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from face_rec.params import AUTOENCODER_HEIGHT, AUTOENCODER_WIDTH

def pca_reconstruction(pca, photo):
    flattened_face = np.expand_dims(photo.reshape(np.prod(photo.shape)), axis=0)
    data_projected = pca.transform(flattened_face)
    reconstructed_photo = pca.inverse_transform(data_projected)[0].reshape(photo.shape)
    return reconstructed_photo

def pca_reconstruction_mean(pca):
    return pca.mean_.reshape((AUTOENCODER_HEIGHT,AUTOENCODER_WIDTH,3))

def pca_reconstruction_main(pca, n_components):
    sc = MinMaxScaler()
    sc = sc.fit(pca.components_)
    input_transformado = sc.transform(pca.components_[0:n_components])
    input_transformado = input_transformado.mean(axis=0)
    return input_transformado.reshape((AUTOENCODER_HEIGHT,AUTOENCODER_WIDTH,3))
