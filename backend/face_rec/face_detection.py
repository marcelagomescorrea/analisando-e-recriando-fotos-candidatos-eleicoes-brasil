import cv2
import numpy as np
import os

AUTOENCODER_WIDTH = int(os.getenv('AUTOENCODER_WIDTH'))
AUTOENCODER_HEIGHT = int(os.getenv('AUTOENCODER_HEIGHT'))

def crop_face(face):
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
  gray = None
  if len(face.shape) == 3:
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
  else:
    gray = face = np.expand_dims(face, axis=-1)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  if len(faces):
    (x,y,w,h) = faces[0]
    w_slack, h_slack = w//2, h//2
    img_cropped = face[
        max(0,y-h_slack):
        min(face.shape[0]-1,y+h+h_slack),
        max(0, x-w_slack):
        min(face.shape[1]-1,x+w+w_slack),
        :]
    return img_cropped
  else:
    square = min(face.shape[0], face.shape[1])//2
    mid_height = face.shape[0]//2
    mid_width = face.shape[1]//2
    return face[mid_height-square:mid_height+square,mid_width-square:mid_width+square]

def resize_face(face, x_max=AUTOENCODER_WIDTH, y_max=AUTOENCODER_HEIGHT):
  scale = min(y_max/face.shape[0], x_max/face.shape[1])
  return cv2.resize(face, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def pad_face(face, x_max=AUTOENCODER_WIDTH, y_max=AUTOENCODER_HEIGHT):
  delta_w = x_max - face.shape[1]
  delta_h = y_max - face.shape[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)
  return cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def gray_face(face):
  return cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

def is_gray(face):
    if len(face.shape) < 3: return True
    if face.shape[2]  == 1: return True
    b,g,r = face[:,:,0], face[:,:,1], face[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False
