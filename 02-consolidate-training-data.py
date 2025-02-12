import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.signal import decimate

directory_to_traverse = "/home/alexbeat/data/kuopio-gait-dataset-processed-v2"  # Change this to your target directory
output_root_path = "/home/alexbeat/data/kuopio-gait-dataset-ml"
input_folder = "input"
output_folder = "output"
fs = 100  # Desired sampling frequency for GRFs

def trim_dataframe(df, multiple):
    """
    Trims the DataFrame from the beginning and end to make the number of frames
    cleanly divisible by a multiple for decimation.

    Parameters:
    df (pd.DataFrame): The DataFrame to be trimmed.

    Returns:
    pd.DataFrame: A new DataFrame with frames trimmed.
    """
    # Step 1: Get the number of frames in the DataFrame
    num_frames = len(df)

    # Step 2: Calculate the number of frames to remove
    frames_to_remove = num_frames % multiple
    print('frames to remove', frames_to_remove)

    # Step 3: Remove frames from the beginning and end
    if frames_to_remove > 0:
        # Calculate how many to remove from the beginning and end
        remove_from_start = frames_to_remove // 2
        remove_from_end = frames_to_remove - remove_from_start
        
        # Slice the DataFrame
        cleaned_df = df.iloc[remove_from_start: num_frames - remove_from_end]
    else:
        cleaned_df = df  # No frames to remove if already divisible by 10

    return cleaned_df

# Function to process each group of files
def process_file_group(top_level_dir, parent_suffix, file_group):
    """Function to process a file based on its extension."""
    print(f"Processing group from {top_level_dir}: {parent_suffix}: {file_group}")

    processed = False
    written = False
    input_df = pd.DataFrame()
    output_df = pd.DataFrame()
    for file_path in file_group:
        df = pd.read_csv(file_path,index_col=0)
        # df = load_data(file_path)
        # Split the file name and extension
        file_name, file_extension = os.path.splitext(file_path)
        file_type = file_name.rsplit('_', 1)[-1]

        if file_type == "grfs":
            # print("GRF File!")
            output_df = df
            processed = True
        elif file_type == "accelerations"  or file_type == "orientations":
            # print("Accelerations or Orientations File!")
            # Check if merged_df is empty
            if input_df.empty:
                # If merged_df is empty, set it to the current DataFrame
                input_df = df
            else:
                # Merge with the existing DataFrame on 'time' column
                input_df = pd.merge(input_df, df, on='time')  # Change 'how' as needed
            processed = True
    if processed:
        # Trim the DataFrame by slicing
        multiple_of = 10
        desired_length = multiple_of * len(input_df)
        output_trimmed_df = df.iloc[:desired_length]  # Keep all rows except the last `num_rows`
        print(input_df.shape)
        print(output_trimmed_df.shape)
        
        input_df_trim = trim_dataframe(input_df, fs)
        output_df_trim = trim_dataframe(output_trimmed_df, fs * multiple_of)
        # Display the cleaned DataFrame and its new length
        print("Input Number of frames:", len(input_df_trim))
        print("Output Number of frames:", len(output_df_trim))

        # downsampling factor
        input_df = input_df_trim.reset_index(drop=True)
        decim_factor = int(fs * multiple_of / fs)
        output_df = output_df_trim.apply(lambda x: decimate(x, decim_factor))
        # print(df.shape)
        # Get the directory path
        # directory_path = os.path.dirname(file_path)


        # Get the file name and extension
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))

        # Display the results
        # print(f"Directory Path: {directory_path}")
        # print(f"File Name: {file_name}")
        # print(f"File Extension: {file_extension}")
        output_dir = os.path.join(output_root_path, top_level_dir)
    
        input_csv_file_path = os.path.join(output_root_path,input_folder,f"{top_level_dir}-{parent_suffix}-input.csv")
        output_csv_file_path = os.path.join(output_root_path,output_folder, f"{top_level_dir}-{parent_suffix}-output.csv")
        # print("shape test")
        # print(input_df.shape)
        # print(output_df.shape)
        # Write the DataFrame to a CSV file
        input_df.to_csv(input_csv_file_path, index=True)  # Write to CSV without the index
        output_df.to_csv(output_csv_file_path, index=True)  # Write to CSV without the index

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
    filtered_files = [file for file in file_paths if file.endswith(".csv")]

    # Create a nested dictionary to hold the grouped files
    grouped_files = defaultdict(lambda: defaultdict(list))

    # Group files by the third to last subdirectory and then by the suffix after the last two underscores
    for path in filtered_files:
        # Split the path and get the third to last subdirectory
        parts = path.split(os.sep)
        if len(parts) > 2:  # Ensure there are at least two subdirectories
            second_to_last_subdirectory = parts[-3]  # Get the third to last subdirectory
            # Get the filename from the path
            filename = os.path.basename(path)

            # Remove "data_" from the filename if it starts with "data_"
            if filename.startswith("data_"):
                filename = filename[5:]
            # Get the suffix after the last two underscores
            parts_of_filename = filename.rsplit('_', 1)  # Split by underscores, limit to the last two
            if len(parts_of_filename) > 1:
                # The key is the last part before the last two underscores
                suffix = parts_of_filename[-2]  # Get the part before the last two underscores
                # suffix = parts_of_filename[-2] + '_' + parts_of_filename[-1].split('.')[0]  # Combine the last two parts
            else:
                suffix = filename.split('.')[0]  # If not enough parts, take the full filename without extension
            # Append the path to the corresponding group
            grouped_files[second_to_last_subdirectory][suffix].append(path)

    # Display the grouped files
    for subdirectory, suffix_group in grouped_files.items():
        print(f"{subdirectory}:")
        for suffix, files in suffix_group.items():
            print(f"  {suffix}: {files}")

        # Create a new list that only includes groups with exactly 3 files
    selected_files = []
    for top_level_dir, key_group in grouped_files.items():
        for key, files in key_group.items():
            if len(files) == 3:  # Check if the number of files is 3
                selected_files.append((top_level_dir,key,files))  # Add the files to the new list
                print(f"  {key}: {files}")

    # Determine the number of available threads
    max_workers = os.cpu_count()  # Get the number of CPU cores
    # max_workers = 1
    print(f"Using {max_workers} threads.")
    # print(filtered_files)
    # Use ThreadPoolExecutor to process each group of three files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for top_level_dir, parent_suffix, file_group in selected_files:
            futures.append(executor.submit(process_file_group, top_level_dir, parent_suffix, file_group))

    # Wait for all futures to complete (optional)
    for future in futures:
        future.result()  # This will raise exceptions if any occurred during processing

    # Print the results (file sizes in this case)
    # for file_path, size in zip(sto_files, results):
    #     print(f"Processed {file_path}: {size}")


if __name__ == "__main__":
    print("---Beginning Script!---")

    os.makedirs(os.path.join(output_root_path,output_folder), exist_ok=True)  # exist_ok=True prevents raising an error if the directory exists
    os.makedirs(os.path.join(output_root_path,input_folder), exist_ok=True) 
    # Specify the directory to traverse
   
    main(directory_to_traverse)
    print("---Ending Script!---")
