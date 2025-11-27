import glob
import os
import re
import json
import argparse
import pandas as pd

def get_relative_path(full_path, base_path):
    """
    Get the relative path of a file from the base path.

    Args:
        full_path (str): The full path of the file.
        base_path (str): The base path to exclude.

    Returns:
        str: The relative path.
    """
    return os.path.relpath(full_path, base_path)

def process_xlsx(file_path):
    """
    Process an Excel file to extract data from the first sheet.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the first sheet.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        # print(f"Processed {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def preprocess_dataframe(df, dirname=None):
    """
    Preprocess a DataFrame to convert it into a list of dictionaries with keys 'name' and 'run_num'.

    Args:
        df (pd.DataFrame): DataFrame with columns as assurance case names and rows labeled as 'run_{num}'.

    Returns:
        List[dict]: List of dictionaries with keys 'name' and 'run_num'.
    """
    result = []
    for run_num in df.index:  # Iterate over rows (index)
        for assurance_case_name in df.columns[1:]:  # Iterate over columns
            result.append({
                "name": assurance_case_name,
                "run_num": run_num,
                "content": df.at[run_num, assurance_case_name],  # Get the value for the specific run and case
                "dirname": dirname,  # Add the dirname to the result
                "doc_id": f"{dirname}_{assurance_case_name}_{run_num}" if dirname else f"{assurance_case_name}_{run_num}"
            })
    return result


def preprocess_files(file_path, base_path):
    # read the txt files and convert them to a dictionary
    result = {}
    with open(file_path, 'r') as f:
        content = f.read()
        result['content'] = content
        # result['name'] = os.path.basename(file_path), name should be before Result/{run_num}/
        # get the folder after base_path before Result/{run_num}/
        # Regex pattern
        pattern = r"/([^/]+)/([^/]+)/Result"

        # Search for the pattern
        match = re.search(pattern, file_path)
        if match:
            result['name'] = f"{match.group(1)}_{match.group(2)}"
            
        # run_num = os.path.basename(file_path).split('_')[-1].split('.')[0]  # Extract run number from filename
        result['run_num'] = os.path.basename(file_path).split('_')[-1].split('.')[0]  # Extract run number from filename
        result['dirname'] = os.path.dirname(file_path).split(os.sep)[-1]  # Get the directory name
        result['doc_id'] = f"{result['dirname']}_{result['name']}_{result['run_num']}"  # Create a unique doc_id

    return result

def iterate_files(base_pattern, extension=".xlsx"):
    """
    Iterate over all folders matching the base pattern and process .xlsx files within them.

    Args:
        base_pattern (str): Glob pattern to match folders.
    """
    # Find all folders matching the pattern
    result_folders = glob.glob(base_pattern)
    # print(f"Found {len(result_folders)} folders matching the pattern: {base_pattern}")

    match_files = []

    for folder in result_folders:
        # Iterate over all files in the folder
        for root, _, files in os.walk(folder):
            # print(f"Processing folder: {root}")
            for file in files:
                if file.endswith(extension):  # Process only files with the specified extension
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}")
                    # create absolute path
                    abs_path = os.path.abspath(file_path)
                    match_files.append(abs_path)

    return match_files

def main(args):
    # Base pattern to match folders
    base_pattern = args.base_path + "/*/"
    print(f"Searching for folders matching: {base_pattern}")

    # Start the iteration
    matched_files = iterate_files(base_pattern, extension=args.extension)

    all_data = []

    if args.extension == ".xlsx":
        # Process each matched Excel file
        for file_path in matched_files:
            # get the file path last dirname
            dirname = os.path.dirname(file_path).split(os.sep)[-1]
            basename = os.path.basename(file_path)
            rel_path = get_relative_path(file_path, args.base_path)
            print(f"Processing file: {dirname}, {basename}, relative path: {rel_path}")
            df = process_xlsx(file_path)
            if df is not None:
                # print(f"Data from {file_path}:\n{df.head()}\n")
                processed_data = preprocess_dataframe(df, dirname=dirname)
                print(f"Processed data: {processed_data[0].keys()}\n, length: {len(processed_data)}")
                all_data.extend(processed_data)
    else:
        # Print the matched files
        print("Matched files:")
        for file in matched_files:
            # print(file)
            processed_data = preprocess_files(file, args.base_path)
            all_data.append(processed_data)

    print(f"Total processed data entries: {len(all_data)}")
    # Save the processed data to a JSON file
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(all_data, f, indent=4)
        print(f"Processed data saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate over files in folders matching a base pattern.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to search for folders")
    parser.add_argument("--extension", "--ext", type=str, default=".xlsx", 
                        help="File extension to filter files (default: .xlsx)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save processed data (default: output.json)")
    args = parser.parse_args()

    main(args)

