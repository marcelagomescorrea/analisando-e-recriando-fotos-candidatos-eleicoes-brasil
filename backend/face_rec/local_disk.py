import cv2
import os
from face_rec.params import LOCAL_DATA_PATH_OUTPUT_IMG

def open_local_image(path):
  img = cv2.imread(path, cv2.COLOR_BGR2RGB)
  return img

def save_local_image(filename: str, face, bw: bool, eleito: bool):
  folder = None

  if bw:
    folder = os.path.join(LOCAL_DATA_PATH_OUTPUT_IMG,
                          'bw', 'elected' if eleito else 'not_elected')
  else:
    folder = os.path.join(LOCAL_DATA_PATH_OUTPUT_IMG,
                          'color', 'elected' if eleito else 'not_elected')

  if not os.path.exists(folder):
    os.makedirs(folder)
  cv2.imwrite(os.path.join(folder, filename), face)

  return None
