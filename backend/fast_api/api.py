from fastapi import FastAPI, UploadFile, File
#from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import cv2
from face_rec.face_detection import crop_face, resize_face, pad_face
from face_rec.face_reconstruction import pca_reconstruction, pca_reconstruction_main
from pca_logic.registry import load_pca
from autoencoder_logic.registry import load_autoencoder
from autoencoder_logic.model import r_loss, kl_loss, total_loss
from autoencoder_logic.main import pred

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

pcas = {}
autoencoders = {}

@app.on_event("startup")
async def startup_event():
    for idx in range(4):
        elected = idx&1 == 1
        bw = idx&2==2
        pcas[(elected, bw)] = load_pca(elected=elected, bw=bw, save_copy_locally=False)
        autoencoders[(elected, bw)] = load_autoencoder(elected=elected, bw=bw, custom_objects={'r_loss': r_loss, 'kl_loss': kl_loss, 'total_loss': total_loss}, save_copy_locally=False)

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    ### Do cool stuff with your image.... For example face detection
    print(cv2_img.shape)
    annotated_img = pad_face(resize_face(crop_face(cv2_img)))

    ### Encoding and responding with the image
    im = cv2.imencode('.png', annotated_img)[1] # extension depends on which format is sent from Streamlit
    return Response(content=im.tobytes(), media_type="image/png")

@app.post('/reconstruct_pca_main_components')
async def receive_pca_mean_request(elected: bool, bw: bool, n_components: int):
    pca = pcas[(elected, bw)]

    annotated_img = pca_reconstruction_main(pca, n_components)*255

    im = cv2.imencode('.png', annotated_img)[1]
    return Response(content=im.tobytes(), media_type="image/png")

@app.post('/reconstruct_pca')
async def receive_pca_image(elected: bool, bw: bool, img: UploadFile=File(...)):
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pca = pcas[(elected, bw)]
    face = pad_face(resize_face(crop_face(cv2_img)))

    annotated_img = pca_reconstruction(pca, face)

    im = cv2.imencode('.png', annotated_img)[1]
    return Response(content=im.tobytes(), media_type="image/png")

@app.post('/reconstruct_autoencoder_randomly')
async def receive_autoencoder_randomly_request(elected: bool, bw: bool):
    autoencoder = autoencoders[(elected, bw)]

    annotated_img = pred(autoencoder, None)[0]*255

    im = cv2.imencode('.png', annotated_img)[1]
    return Response(content=im.tobytes(), media_type="image/png")

@app.post('/reconstruct_autoencoder')
async def receive_autoencoder_image(elected: bool, bw: bool, img: UploadFile=File(...)):
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    autoencoder = autoencoders[(elected, bw)]
    face = pad_face(resize_face(crop_face(cv2_img)))

    annotated_img = pred(autoencoder, np.expand_dims(face, axis=0))[0]*255

    im = cv2.imencode('.png', annotated_img)[1]
    return Response(content=im.tobytes(), media_type="image/png")
