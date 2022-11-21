import os

def get_img_filename(year: str, state: str, sq_candidato: str):

  path = os.path.join(os.path.expanduser(os.environ.get("LOCAL_DATA_PATH_INPUT_IMG")), year)

  options = ['F'+state+str(sq_candidato)+'_div.jpg', 'F'+state+str(sq_candidato)+'_div.jpeg',
             'F'+state+str(sq_candidato)+'.jpg', 'F'+state+str(sq_candidato)+'.jpeg']

  for option in options:
    full_path = os.path.join(path, option)
    if os.path.exists(full_path): return full_path

  return None
