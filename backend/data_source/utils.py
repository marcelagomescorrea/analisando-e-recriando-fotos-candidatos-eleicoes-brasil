import os
from data_source.params import LOCAL_DATA_PATH_INPUT_IMG

def get_img_filename(year: str, state: str, sq_candidato: str):

  path = os.path.join(LOCAL_DATA_PATH_INPUT_IMG, year)
  options = ['F'+state+sq_candidato+'_div.jpg', 'F'+state+sq_candidato+'_div.jpeg',
             'F'+state+sq_candidato+'.jpg', 'F'+state+sq_candidato+'.jpeg']

  for option in options:
    full_path = os.path.join(path, option)
    if os.path.exists(full_path): return full_path

  return None
