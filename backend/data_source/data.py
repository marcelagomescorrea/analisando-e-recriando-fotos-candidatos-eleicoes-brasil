import pandas as pd
from data_source.local_disk import get_pandas_chunk, save_local_chunk, extract_local_files
from data_source.params import COLUMN_NAMES_FULL

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by removing buggy or irrelevant transactions
    or columns for the training set
    """

    # remove useless/redundant columns
    df = df[COLUMN_NAMES_FULL]
    df = df.dropna(subset=['filename'])

    print("✅ data cleaned")

    return df

def get_chunk(year: str,
              state: str,
              index: int = 0,
              chunk_size: int = None,
              verbose=False) -> pd.DataFrame:
    """
    Return a `chunk_size` rows from the source dataset, starting at row `index` (included)
    Always assumes `source_name` (CSV or Big Query table) have headers,
    and do not consider them as part of the data `index` count.
    """

    # if os.environ.get("DATA_SOURCE") == "big query":

    #     chunk_df = get_bq_chunk(table=source_name,
    #                             index=index,
    #                             chunk_size=chunk_size,
    #                             dtypes=dtypes,
    #                             verbose=verbose)

    #     return chunk_df

    chunk_df = get_pandas_chunk(year=year,
                                state=state,
                                index=index,
                                chunk_size=chunk_size,
                                verbose=verbose)

    return chunk_df

def save_chunk(data: pd.DataFrame) -> None:
    """
    save chunk
    """

    # if os.environ.get("DATA_SOURCE") == "big query":

    #     save_bq_chunk(table=destination_name,
    #                   data=data,
    #                   is_first=is_first)

    #     return

    save_local_chunk(data=data)

def extract_files() -> dict:
    """
    extract files
    """

    # if os.environ.get("DATA_SOURCE") == "big query":
    #     ...
    #     return

    return extract_local_files()
