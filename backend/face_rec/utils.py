import cv2
import os

def open_image_local(path):
  img = cv2.imread(path, cv2.COLOR_BGR2RGB)
  return img

def save_image_local(filename: str, face, bw: bool, eleito: bool):
  folder = None

  if bw:
    folder = os.path.join(os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_OUTPUT_IMG")),
                          'bw', 'elected' if eleito else 'not_elected')
  else:
    folder = os.path.join(os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_OUTPUT_IMG")),
                          'color', 'elected' if eleito else 'not_elected')


  if not os.path.exists(folder):
    os.makedirs(folder)
  cv2.imwrite(os.path.join(folder, filename), face)

  return None
