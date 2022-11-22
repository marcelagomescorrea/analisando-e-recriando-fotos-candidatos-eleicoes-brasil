import re
import os
import pandas as pd
import numpy as np
import zipfile

from data_source.utils import get_img_filename
from data_source.params import COLUMN_NAMES, \
                                states, \
                                ELEITO, \
                                NAO_ELEITO, \
                                LOCAL_DATA_PATH_CSV, \
                                LOCAL_DATA_PATH_SRC, \
                                FILENAME_COLUMN_NAME, \
                                FACE_COLUMN_NAME, \
                                ID_COLUMN_NAME, \
                                LOCAL_DATA_PATH_INPUT_IMG, \
                                STATE_COLUMN_NAME, \
                                YEAR_COLUMN_NAME, \
                                ELLECTED_COLUMN_NAME

from face_rec.face_detection import gray_face, is_gray, pad_face, resize_face, crop_face
from face_rec.local_disk import save_local_image, open_local_image

def get_pandas_chunk(year: str,
                     state: str,
                     index: int,
                     chunk_size: int,
                     verbose=True) -> pd.DataFrame:
    """
    return a chunk of the raw dataset from local disk or cloud storage
    """

    full_path = os.path.join(
        LOCAL_DATA_PATH_CSV,
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


        df[FILENAME_COLUMN_NAME[0]] = df[ID_COLUMN_NAME[0]].map(lambda id_candidato: get_img_filename(year, state, str(id_candidato)))

    except pd.errors.EmptyDataError:
        return None  # end of data

    return df

def save_local_chunk(data: pd.DataFrame):
    """
    save a chunk of the dataset to local disk
    """

    for ind in data.index:
        state = data[STATE_COLUMN_NAME[0]][ind]
        year = str(data[YEAR_COLUMN_NAME[0]][ind])
        eleito = data[ELLECTED_COLUMN_NAME[0]][ind]

        if eleito not in (ELEITO + NAO_ELEITO):
            continue

        sq_candidato = str(data[ID_COLUMN_NAME[0]][ind])
        face = data[FACE_COLUMN_NAME[0]][ind]

        if is_gray(face):
            save_local_image(year+'F'+state+str(sq_candidato)+'_div.jpg', face, True, eleito in ELEITO)
        else:
            save_local_image(year+'F'+state+str(sq_candidato)+'_div.jpg', face, False, eleito in ELEITO)
            save_local_image(year+'F'+state+str(sq_candidato)+'_div.jpg', gray_face(face), True, eleito in ELEITO)


def extract_local_files() -> dict:
    def unzip_local_files(year: str, filename: str, csv: bool):
        from_path = os.path.join(
            LOCAL_DATA_PATH_SRC,
            year,
            filename)

        zip_ref = zipfile.ZipFile(from_path, 'r')

        to_path = os.path.join(
            LOCAL_DATA_PATH_CSV if csv else LOCAL_DATA_PATH_INPUT_IMG,
            year)

        #zip_ref.extractall(to_path)
        zip_ref.close()

        if csv:
            states_found = []
            state_str = '|'.join(states)
            extracted_filenames = os.listdir(to_path)
            for extracted_filename in extracted_filenames:
                match = re.match(rf'.*({state_str}).csv$', extracted_filename)
                if match is not None:
                    states_found.append(match.group(1))
                    print(f"{year}: ✅ found state {match.group(1)} to preprocess 👌")
            return states_found

    years = os.listdir(LOCAL_DATA_PATH_SRC)
    result = dict()
    for year in years:
        match = re.match(r'(\d+)', year)
        if match is not None:
            print(f"✅ found year {match.group(1)} to preprocess 👌")
            src_year_folder = os.path.join(LOCAL_DATA_PATH_SRC, year)
            zipped_files = os.listdir(src_year_folder)
            for zipped_file in zipped_files:
                if zipped_file.startswith('consulta') and zipped_file.endswith('.zip'):
                    result[year] = unzip_local_files(year, zipped_file, csv=True)
                elif zipped_file.startswith('foto') and zipped_file.endswith('.zip'):
                    unzip_local_files(year, zipped_file, csv=False)
    return result

def load_local_chunk_images(df: pd.DataFrame) -> np.ndarray:
  def open_local_images(filename: str) -> list:
    return pad_face(resize_face(crop_face(open_local_image(filename))))
  df[FACE_COLUMN_NAME[0]] = df[FILENAME_COLUMN_NAME[0]].map(open_local_images)
  return df
