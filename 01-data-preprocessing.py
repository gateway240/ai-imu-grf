import os
import concurrent.futures
import pandas as pd
import numpy as np
from scipy.signal import decimate

directory_to_traverse = "/home/alexbeat/data/kuopio-gait-dataset-processed-v2"  # Change this to your target directory

def load_data(file_path):
    df = pd.read_csv(file_path, skiprows=[1, 2, 3, 4], header=1, delimiter="\t")
    return df

def process_file(file_path):
    """Function to process a file based on its extension."""
    print(f"Processing file: {file_path}")

    processed = False
    written = False
    df = load_data(file_path)
    # Split the file name and extension
    file_name, file_extension = os.path.splitext(file_path)
    file_type = file_name.rsplit('_', 1)[-1]

    result_df = []
    if file_type == "grfs":
        # print("GRF File!")
        force_num = 2
        forces_df = df[
            ["time", f"f{force_num}_1", f"f{force_num}_2", f"f{force_num}_3", f"p{force_num}_1", f"p{force_num}_2", f"p{force_num}_3", f"m{force_num}_1", f"m{force_num}_2", f"m{force_num}_3"]
        ]
        # Originally had down sampling/decimating here but need to move it to later in processing
        # This is needed since if the original samples are not a multiple of 10 it won't decimate cleanly

        result_df = forces_df
        processed = True
    elif file_type == "accelerations"  or file_type == "orientations":
        # print("Accelerations or Orientations File!")
        split_df = pd.concat([df[col].str.split(",", expand=True).add_prefix(f"{col}_{file_type[:3]}_") for col in df.columns[1:]], axis=1)
        result_df = pd.concat([df[["time"]], split_df], axis=1)
        processed = True
    if processed:
        # print(df.shape)
        # Get the directory path
        directory_path = os.path.dirname(file_path)

        # Get the file name and extension
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))

        # Display the results
        # print(f"Directory Path: {directory_path}")
        # print(f"File Name: {file_name}")
        # print(f"File Extension: {file_extension}")
        csv_file_path = os.path.join(directory_path,f"{file_name}.csv")
        # Write the DataFrame to a CSV file
        result_df.to_csv(csv_file_path, index=True)  # Write to CSV without the index

        # print(f"DataFrame written to {csv_file_path}")
    return processed and written


def traverse_directory(directory):
    """Recursively traverse the directory and return a list of file paths."""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def main(directory):
    """Main function to traverse the directory and process files using a task pool."""
    # Get all file paths in the directory
    file_paths = traverse_directory(directory)
    sto_files = [file for file in file_paths if file.endswith(".sto")]

    # Determine the number of available threads
    max_workers = os.cpu_count()  # Get the number of CPU cores
    print(f"Using {max_workers} threads.")

    # Use a thread pool to process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the process_file function to the file paths
        results = list(executor.map(process_file, sto_files))

    # Print the results (file sizes in this case)
    # for file_path, size in zip(sto_files, results):
    #     print(f"Processed {file_path}: {size}")


if __name__ == "__main__":
    print("---Beginning Script!---")
    # Specify the directory to traverse
    main(directory_to_traverse)
    print("---Ending Script!---")
