import zipfile
import os
import pandas as pd
import numpy as np
import cv2

from data_source.utils import get_img_filename
from face_rec.face_detection import gray_face, is_gray
from data_source.params import COLUMN_NAMES

def open_image_local(path: str):
  img = cv2.imread(path, cv2.COLOR_BGR2RGB)
  return img

def get_pandas_chunk(year: str,
                     state: str,
                     index: int,
                     chunk_size: int,
                     verbose=True) -> pd.DataFrame:
    """
    return a chunk of the raw dataset from local disk or cloud storage
    """

    full_path = os.path.join(
        os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_CSV")),
        year,
        f"consulta_cand_{year}_{state}.csv")

    if verbose:
        print(f"Source data from {full_path}: {chunk_size if chunk_size is not None else 'all'} rows (from row {index})")

    try:
        df = pd.read_csv(
                full_path,
                skiprows=np.arange(1, index+1),  # skip header
                nrows=chunk_size,
                header=0,
                encoding='iso-8859-1',
                on_bad_lines='warn',
                sep=';',
                usecols=COLUMN_NAMES)  # read all rows


        df['filename'] = df['SQ_CANDIDATO'].map(lambda sq_candidato: get_img_filename(year, state, sq_candidato))

    except pd.errors.EmptyDataError:
        return None  # end of data

    return df

def save_image_local(filename: str, face, bw: bool, eleito: bool):
  folder = None

  if bw:
    folder = os.path.join(os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_OUTPUT_IMG")),
                          'elected' if eleito else 'not_elected',
                          'bw')
  else:
    folder = os.path.join(os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_OUTPUT_IMG")),
                          'elected' if eleito else 'not_elected',
                          'color')


  if not os.path.exists(folder):
    os.makedirs(folder)
  cv2.imwrite(os.path.join(folder, filename), face)

  return None

def save_local_chunk(data: pd.DataFrame):
  """
  save a chunk of the dataset to local disk
  """

  for ind in data.index:
    state = data['SG_UE'][ind]
    year = str(data['ANO_ELEICAO'][ind])
    eleito = data['DS_SIT_TOT_TURNO'][ind]

    if eleito == "ELEITO" or eleito == "ELEITO POR MÉDIA":
      eleito = True
    elif eleito == "NÃO ELEITO":
      eleito = False
    else:
      continue

    sq_candidato = str(data['SQ_CANDIDATO'][ind])
    face = data['face'][ind]

    if is_gray(face):
      save_image_local(year+'F'+state+str(sq_candidato)+'_div.jpg', face, True, eleito)
    else:
      save_image_local(year+'F'+state+str(sq_candidato)+'_div.jpg', face, False, eleito)
      save_image_local(year+'F'+state+str(sq_candidato)+'_div.jpg', gray_face(face), True, eleito)


def extract_files(year: str, filename: str, csv: bool):
    from_path = os.path.join(
        os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_SRC")),
        year,
        filename)

    zip_ref = zipfile.ZipFile(from_path, 'r')

    zip_ref.extractall(os.path.join(
        os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_CSV")) if csv else os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_INPUT_IMG")),
        year))
    zip_ref.close()
