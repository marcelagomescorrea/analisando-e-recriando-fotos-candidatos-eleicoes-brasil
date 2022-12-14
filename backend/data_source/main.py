from data_source.data import clean_data, get_chunk, save_chunk, extract_files, load_chunk_images
from data_source.params import CHUNK_SIZE

def preprocess():
    year_and_states_list = extract_files()
    all_chunk_count = 0
    all_rows_count = 0
    all_cleaned_rows_count = 0

    for year, states_list in year_and_states_list.items():
        year_chunk_count = 0
        year_rows_count = 0
        year_cleaned_rows_count = 0

        for state in states_list:

            # iterate on the dataset, by chunks
            chunk_id = 0
            row_count = 0
            cleaned_row_count = 0

            while (True):

                print(f"\n{year}, {state}: Processing chunk nĀ°{chunk_id}...")

                data_chunk = get_chunk(year=year,
                                    state=state,
                                    index=chunk_id * CHUNK_SIZE,
                                    chunk_size=CHUNK_SIZE)

                # Break out of while loop if data is none
                if data_chunk is None:
                    print(f"{year}, {state}: No data in latest chunk...")
                    break

                row_count += data_chunk.shape[0]

                data_chunk_cleaned = clean_data(data_chunk)

                cleaned_row_count += len(data_chunk_cleaned)

                # break out of while loop if cleaning removed all rows
                if len(data_chunk_cleaned) == 0:
                    print(f"{year}, {state}: ā No cleaned data in latest chunk...")
                    break
                else:
                    print(f"{year}, {state}: ā data cleaned")

                images_processed_chunk = load_chunk_images(data_chunk_cleaned)

                save_chunk(images_processed_chunk)

                chunk_id += 1

            if row_count == 0:
                print(f"{year}, {state}: ā no new data for the preprocessing š")
                break

            print(f"{year}, {state}: ā data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")
            year_chunk_count += chunk_id
            year_rows_count += row_count
            year_cleaned_rows_count += cleaned_row_count

        print(f"{year}: ā data processed saved entirely: {year_chunk_count} chunks {year_rows_count} rows ({year_cleaned_rows_count} cleaned)")
        all_chunk_count += year_chunk_count
        all_rows_count += year_rows_count
        all_cleaned_rows_count += year_cleaned_rows_count

    print(f"ā data processed saved entirely: {all_chunk_count} chunks {all_rows_count} rows ({all_cleaned_rows_count} cleaned)")
    return None
