import cv2
import numpy as np
from dotenv import load_dotenv
import os

AUTOENCODER_WIDTH = int(os.getenv('AUTOENCODER_WIDTH'))
AUTOENCODER_HEIGHT = int(os.getenv('AUTOENCODER_HEIGHT'))

def crop(face):
  face_cascade = cv2.CascadeClassifier('face_rec/haarcascade_frontalface_default.xml')

  faces = face_cascade.detectMultiScale(face, 1.1, 4)
  if len(faces):
    (x,y,w,h) = faces[0]
    img_cropped = face[
        max(0,y-30):
        min(face.shape[0]-1,y+h+30),
        max(0, x-30):
        min(face.shape[1]-1,x+w+30),
        :]
    return img_cropped
  else:
    square = min(face.shape[0], face.shape[1])//2
    mid_height = face.shape[0]//2
    mid_width = face.shape[1]//2
    return face[mid_height-square:mid_height+square,mid_width-square:mid_width+square]

def pad(face, x_max=AUTOENCODER_WIDTH, y_max=AUTOENCODER_HEIGHT):
  scale = min(y_max/face.shape[0], x_max/face.shape[1])
  resized_face = cv2.resize(face, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

  delta_w = x_max - resized_face.shape[1]
  delta_h = y_max - resized_face.shape[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)

  return cv2.copyMakeBorder(resized_face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
