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

@app.post('/reconstruct_random')
async def receive_pca_mean_request(model: str, elected: bool, bw: bool, n_components: int):
    loaded_model = pcas[(elected, bw)] if model == 'pca' else autoencoders[(elected, bw)]
    annotated_img = pca_reconstruction_main(loaded_model, n_components)*255 if model == 'pca' else pred(loaded_model, None)[0]*255

    im = cv2.imencode('.png', annotated_img)[1]
    return Response(content=im.tobytes(), media_type="image/png")

@app.post('/reconstruct_photo')
async def receive_pca_image(model: str, elected: bool, bw: bool, img: UploadFile=File(...)):
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    face = pad_face(resize_face(crop_face(cv2_img)))

    loaded_model = pcas[(elected, bw)] if model == 'pca' else autoencoders[(elected, bw)]

    annotated_img = pca_reconstruction(loaded_model, face) if model == 'pca' else pred(loaded_model, np.expand_dims(face, axis=0))[0]*255

    im = cv2.imencode('.png', annotated_img)[1]
    return Response(content=im.tobytes(), media_type="image/png")
