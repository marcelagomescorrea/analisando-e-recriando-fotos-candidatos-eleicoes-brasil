import re
import os
import pandas as pd
import numpy as np
import zipfile

from data_source.utils import get_img_filename
from face_rec.face_detection import gray_face, is_gray
from face_rec.utils import save_image_local
from data_source.params import COLUMN_NAMES, states


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


def extract_local_files() -> dict:
    def unzip_local_files(year: str, filename: str, csv: bool):
        from_path = os.path.join(
            os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_SRC")),
            year,
            filename)

        zip_ref = zipfile.ZipFile(from_path, 'r')

        to_path = os.path.join(
            os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_CSV")) if csv else os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_INPUT_IMG")),
            year)

        zip_ref.extractall(to_path)
        zip_ref.close()

        if csv:
            states_found = []
            state_str = '|'.join(states)
            extracted_filenames = os.listdir(to_path)
            for extracted_filename in extracted_filenames:
                match = re.match(rf'.*_({state_str}).csv$', extracted_filename)
                if match is not None:
                    states_found.append(match.group(1))
                    print(f"\n✅ found state {match.group(1)} to preprocess 👌")
            return states_found

    src_folder = os.path.join(os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_SRC")))
    years = os.listdir(src_folder)
    result = dict()
    for year in years:
        match = re.match(r'(\d+)', year)
        if match is not None:
            print(f"\n✅ found year {match.group(1)} to preprocess 👌")
            src_year_folder = os.path.join(src_folder, year)
            zipped_files = os.listdir(src_year_folder)
            for zipped_file in zipped_files:
                if zipped_file.startswith('consulta'):
                    result[year] = unzip_local_files(year, zipped_file, csv=True)
                elif zipped_file.startswith('foto'):
                    unzip_local_files(year, zipped_file, csv=False)
    return result
