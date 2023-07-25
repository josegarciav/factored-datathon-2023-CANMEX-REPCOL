"""
Module for the data ingestion part of the project.
"""

# Importing standard libraries
import pandas as pd
import gzip
import json

def download_and_create_df(file_list: list, 
                           container_client) -> pd.DataFrame:
    """
    Method to download the data from the blob storage.

    Args:
        file_list (list): A list containing zip downloadable paths for a single desired dataframe.

    Returns:
        pd.DataFrame: The general concatenated dataframe containing all the downloaded files.
    """

    dataframes = []
    for file_name in file_list:
        # Download the blob to a stream
        blob_client = container_client.get_blob_client(file_name)
        downloaded_blob = blob_client.download_blob().readall()
        
        # Decompress the gzipped data and read it into a DataFrame
        json_data = gzip.decompress(downloaded_blob).decode('utf-8')
        df = pd.read_json(json_data, lines=True)
        
        # Append the DataFrame to the list of DataFrames
        dataframes.append(df)
        
    df_concat = pd.concat(dataframes, ignore_index=True)
    return df_concat


