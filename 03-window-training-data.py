import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.signal import decimate

directory_to_traverse = "/home/alexbeat/data/kuopio-gait-dataset-ml"  # Change this to your target directory
output_root_path = "/home/alexbeat/data/kuopio-gait-dataset-ml"
z_force_key = "f2_3"

# Function to process each group of files
def process_file_group(parent_suffix, file_group):
    """Function to process a file based on its extension."""
    print(f"Processing group from {parent_suffix}: {file_group}")

    processed = False
    input_df = pd.DataFrame()
    output_df = pd.DataFrame()
    result_input_df = pd.DataFrame()
    result_output_df = pd.DataFrame()
    for file_path in file_group:
        df = pd.read_csv(file_path,index_col=0)
        # Split the file name and extension
        file_name, file_extension = os.path.splitext(file_path)
        if file_name.endswith("input"):
            input_df = df
            processed = True
        elif file_name.endswith("output"):
            output_df = df
            processed = True
    if processed:
        # print(f'{input_df.shape} & {output_df.shape}')
        if (len(input_df) == len(output_df)):
            #use the prominent component of force to determine activation of the force plate
            index_z = np.where(np.abs(output_df[z_force_key])>5)[0]
            if index_z.size > 0:
                min_index = index_z.min()
                #max_index = min_index + 100 
                #max_index = index_z.max()
                # just the force plate activation region
                breaks = np.where(np.diff(index_z) > 5)[0] 
                if breaks.size > 0: 
                    max_index = index_z[breaks[0]]
                else:
                    max_index = index_z.max()
                
                #half of force plate activation region added to both ends of the window
                min_index = min_index-int((max_index-min_index)/2)
                max_index = max_index+int((max_index-min_index)/2) 
                
                print(f"min: {min_index} and max: {max_index}")
                #extended window length

                windowed_out = output_df.iloc[min_index:max_index + 1]
                windowed_in = input_df.iloc[min_index:max_index + 1]
                # print(f'test: {windowed_out.head()}')
                # print("Input Number of frames:", len(windowed_out))
                # print("Output Number of frames:", len(windowed_in))
                result_input_df = windowed_in
                result_output_df = windowed_out
            else:
               print(f'[Warning] no force data found: index_z == 0') 
        else:
            print(f'[Warning] length mismatch: {len(input_df)} != {len(output_df)}')
 
    return (result_input_df, result_output_df)


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
    grouped_files = defaultdict(list)

    # Group files by the third to last subdirectory and then by the suffix after the last two underscores
    for path in filtered_files:
        # Split the path and get the third to last subdirectory
        # Get the filename from the path
        filename = os.path.basename(path)

        # Get the suffix after the last two underscores
        parts_of_filename = filename.rsplit('-', 1)  # Split by underscores, limit to the last two
        if len(parts_of_filename) > 1:
            # The key is the last part before the last two underscores
            suffix = parts_of_filename[-2]  # Get the part before the last two underscores
        else:
            suffix = filename.split('.')[0]  # If not enough parts, take the full filename without extension
        # Append the path to the corresponding group
        grouped_files[suffix].append(path)

        # Create a new list that only includes groups with exactly 3 files
    selected_files = []
    for subdirectory, files in grouped_files.items():
        if len(files) == 2:  # Check if the number of files is 3
            selected_files.append((subdirectory,files))  # Add the files to the new list
            print(f"  {subdirectory}: {files}")

    # Determine the number of available threads
    max_workers = os.cpu_count()  # Get the number of CPU cores
    # max_workers = 1
    print(f"Using {max_workers} threads.")
    # print(filtered_files)
    # Use ThreadPoolExecutor to process each group of three files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for parent_suffix, file_group in selected_files:
            futures.append(executor.submit(process_file_group, parent_suffix, file_group))

    # Wait for all futures to complete (optional)
    for future in futures:
        future.result()  # This will raise exceptions if any occurred during processing

    # Print the results (file sizes in this case)
    df_inputs = []
    df_outputs = []
    for file_path, df_result in zip(selected_files, futures):
        print(f"Processed {file_path}")
        tmp_in_df, tmp_out_df = df_result.result()
        df_inputs.append(tmp_in_df)
        df_outputs.append(tmp_out_df)
    # Concatenate all first DataFrames into one large DataFrame
    input_df = pd.concat(df_inputs, ignore_index=True)

    # Concatenate all second DataFrames into another large DataFrame
    output_df = pd.concat(df_outputs, ignore_index=True)

    input_csv_file_path = os.path.join(output_root_path,f"_main-input.csv")
    output_csv_file_path = os.path.join(output_root_path,f"_main-output.csv")
    print("shape test")
    print(input_df.shape)
    print(output_df.shape)
    # Write the DataFrame to a CSV file
    input_df.to_csv(input_csv_file_path, index=True)  # Write to CSV without the index
    output_df.to_csv(output_csv_file_path, index=True)  # Write to CSV without the index


if __name__ == "__main__":
    print("---Beginning Script!---")

    # os.makedirs(os.path.join(output_root_path,output_folder), exist_ok=True)  # exist_ok=True prevents raising an error if the directory exists
    os.makedirs(os.path.join(output_root_path), exist_ok=True) 
    # Specify the directory to traverse
   
    main(directory_to_traverse)
    print("---Ending Script!---")
