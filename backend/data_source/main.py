import os
import numpy as np
import pandas as pd

from data_source.data import clean_data, get_chunk, save_chunk, unzip_files
from data_source.params import states
from face_rec.face_detection import pad_face, resize_face, crop_face, open_image_local

def preprocess_chunk_images(year: str, state: str, df: pd.DataFrame) -> np.ndarray:
  def preprocess_image(filename: str) -> list:
    return pad_face(resize_face(crop_face(open_image_local(filename))))
  df['face'] = df['filename'].map(preprocess_image)
  return df

def preprocess_images(year: str, state: str):

    # iterate on the dataset, by chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))

    while (True):

        print(f"\n{year}, {state}: Processing chunk n°{chunk_id}...")

        data_chunk = get_chunk(year=year,
                               state=state,
                               index=chunk_id * CHUNK_SIZE,
                               chunk_size=CHUNK_SIZE)

        # Break out of while loop if data is none
        if data_chunk is None:
            print("No data in latest chunk...")
            break

        row_count += data_chunk.shape[0]

        data_chunk_cleaned = clean_data(data_chunk)

        cleaned_row_count += len(data_chunk_cleaned)

        # break out of while loop if cleaning removed all rows
        if len(data_chunk_cleaned) == 0:
            print("\nNo cleaned data in latest chunk...")
            break

        images_processed_chunk = preprocess_chunk_images(year, state, data_chunk_cleaned)

        save_chunk(images_processed_chunk)

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the preprocessing 👌")
        return None

    print(f"\n✅ data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None


def preprocess():
    for year in ['2014', '2018', '2022']:
        unzip_files(year)
        for state in states:
            if year != '2018' and state != 'DF':
                preprocess_images(year, state)
