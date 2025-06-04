import os
import scipy.io # Added for .mat file handling
import numpy as np # Import numpy for unique function

def read_summe_gt_files(base_path=".", target_filename=None):
    """
    Reads and lists files and their content from the summe/GT directory.
    Handles .mat files using scipy.io.loadmat and attempts to read others as text.

    Args:
        base_path (str): The base path where the 'summe/GT' directory is located.
                         Defaults to the current directory.
        target_filename (str, optional): The specific filename to read. 
                                         If None, reads all files in the directory.
                                         Defaults to None.
    Returns:
        dict: A dictionary where keys are filenames and values are their content
              (loaded .mat data or string content).
              Returns an empty dictionary if the directory/file is not found or on error.
    """
    gt_path = os.path.join(base_path, "summe", "GT")
    
    if not os.path.exists(gt_path):
        print(f"Error: Directory not found at {gt_path}")
        return {}
        
    if not os.path.isdir(gt_path):
        print(f"Error: {gt_path} is not a directory.")
        return {}

    file_contents = {}
    
    files_to_process = []
    if target_filename:
        # Check if the target file exists in the directory
        if os.path.exists(os.path.join(gt_path, target_filename)):
            files_to_process.append(target_filename)
        else:
            print(f"Error: Target file '{target_filename}' not found in {gt_path}")
            return {}
    else:
        try:
            files_to_process = os.listdir(gt_path)
        except Exception as e:
            print(f"An error occurred while listing files: {e}")
            return {}

    print(f"Reading from {gt_path}:")
    for file_name in files_to_process:
        file_path = os.path.join(gt_path, file_name)
        if os.path.isfile(file_path): # Ensure it's a file
            try:
                if file_name.lower().endswith(".mat"):
                    # Handle .mat files
                    mat_data = scipy.io.loadmat(file_path)
                    file_contents[file_name] = mat_data
                    print(f"\n--- Content of {file_name} (MATLAB file) ---")
                    # MAT data is a dict, print keys or a summary
                    print(f"Keys in .mat file: {list(mat_data.keys())}")
                    
                    # Specifically look for and print user_score and gt_score
                    # user_score: Typically a 2D numpy array (frames x annotators).
                    #             Values represent frame-level importance scores from human annotators.
                    #             For the SumMe dataset, these are usually binary (0 = not selected, 1 = selected by the annotator).
                    #             Other datasets might use integer scales or floats, but SumMe GT is typically binary per annotator.
                    # gt_score:   Typically a 1D numpy array (frames x 1).
                    #             Represents the ground truth importance score for each frame.
                    #             Values are often binary (0 or 1), indicating if a frame is
                    #             part of the consensus summary (e.g., selected by a majority
                    #             of annotators), or can be continuous (e.g., average of user_scores
                    #             normalized between 0 and 1).
                    for key_to_check in ['user_score', 'gt_score']:
                        if key_to_check in mat_data:
                            score_data = mat_data[key_to_check]
                            print(f"  Variable '{key_to_check}':")
                            print(f"    Shape: {getattr(score_data, 'shape', 'N/A')}")
                            
                            # Enhanced sample printing and unique value check
                            if hasattr(score_data, 'shape') and score_data.ndim > 0: # Check if it's a numpy array-like object
                                if key_to_check == 'user_score':
                                    unique_values = np.unique(score_data)
                                    print(f"    Unique values in '{key_to_check}': {unique_values}")
                                    if score_data.ndim == 2:
                                        rows = min(score_data.shape[0], 3) # Show up to first 3 rows
                                        cols = min(score_data.shape[1], 3) # Show up to first 3 columns
                                        print(f"    Sample values (up to first {rows}x{cols} slice):")
                                        print(f"{score_data[:rows, :cols]}")
                                    else: # If user_score is not 2D for some reason
                                        elements = min(score_data.size, 10)
                                        print(f"    Sample values (first {elements} elements, flattened):")
                                        print(f"{score_data.flatten()[:elements]}")

                                elif (key_to_check == 'gt_score' or score_data.ndim == 1 or \
                                     (score_data.ndim == 2 and (score_data.shape[0] == 1 or score_data.shape[1] == 1))):
                                    # For gt_score or 1D arrays
                                    unique_values_gt = np.unique(score_data)
                                    print(f"    Unique values in '{key_to_check}': {unique_values_gt}")
                                    elements = min(score_data.size, 10) # Show up to 10 elements
                                    print(f"    Sample values (first {elements} elements, flattened):")
                                    print(f"{score_data.flatten()[:elements]}")
                                else: # Fallback for other multi-dimensional arrays
                                    unique_values_other = np.unique(score_data)
                                    print(f"    Unique values in '{key_to_check}': {unique_values_other}")
                                    elements = min(score_data.size, 5) 
                                    print(f"    Sample values (first {elements} elements, flattened):")
                                    print(f"{score_data.flatten()[:elements]}")
                            elif isinstance(score_data, (list, tuple)) and len(score_data) > 0:
                                try: # Attempt to convert to numpy array for unique, then print list sample
                                    unique_values_list = np.unique(np.array(score_data))
                                    print(f"    Unique values in '{key_to_check}': {unique_values_list}")
                                except: # If conversion fails, just print list sample
                                    print(f"    Could not determine unique values for list/tuple directly.")
                                elements = min(len(score_data), 5)
                                print(f"    Sample values (first {elements} elements):")
                                print(f"{score_data[:elements]}")
                            else: # For scalar or other non-sequence types
                                print(f"    Value: {score_data}")
                        else:
                            print(f"  Variable '{key_to_check}' not found.")

                    # Print shapes of other variables
                    for key, value in mat_data.items():
                        if not key.startswith("__") and key not in ['user_score', 'gt_score']: # Exclude already printed and internal vars
                            print(f"  Variable '{key}' shape: {getattr(value, 'shape', 'N/A')}")
                    print("--- End of .mat content summary ---")
                else:
                    # Attempt to read as text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_contents[file_name] = content
                        print(f"\n--- Content of {file_name} ---")
                        print(content)
                        print("--- End of content ---")
            except UnicodeDecodeError:
                print(f"Error reading file {file_name}: Not a UTF-8 text file. Skipping content.")
                file_contents[file_name] = f"Error: Could not decode as UTF-8 text."
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                file_contents[file_name] = f"Error: {e}"
        elif target_filename and file_name == target_filename: # If target was specified but it's not a file
            print(f"Error: Specified target '{target_filename}' is not a file.")
            return {target_filename: "Error: Not a file."} # Return specific error for target
        elif not target_filename: # Only print skip message if processing all files
             print(f"Skipping {file_name}, as it is not a file.")
    return file_contents

if __name__ == "__main__":
    # Assuming the script is run from d:\Workspace\Classes\TriTueNhanTao\vsumm-reinforce_re
    # and the 'summe/GT' directory is relative to this location.
    # If 'summe' is in a different location, adjust the path accordingly.
    # For example, if 'summe' is in the parent directory: read_summe_gt_files("..")
    # Or provide an absolute path if necessary.
    
    # For this example, let's assume 'summe/GT' is directly inside the project root.
    # If your 'summe' folder is located elsewhere, you might need to adjust this path.
    # For instance, if 'summe' is at 'd:\Workspace\Classes\TriTueNhanTao\vsumm-reinforce_re\data\summe'
    # then you would call: read_summe_gt_files("data")
    
    # Defaulting to look for 'summe/GT' in the same directory as the script.
    # If your 'summe' directory is at the root of 'vsumm-reinforce_re', this should work.
    # If 'summe' is inside another folder, e.g., 'datasets/summe/GT',
    # you would call read_summe_gt_files("datasets")
    
    # Let's assume the 'summe' directory is at the root of the project.
    # So, if this script is in 'd:\Workspace\Classes\TriTueNhanTao\vsumm-reinforce_re\read_gt_data.py',
    # and the GT files are in 'd:\Workspace\Classes\TriTueNhanTao\vsumm-reinforce_re\summe\GT',
    # then the default base_path="." is correct.
    
    project_root = os.path.dirname(os.path.abspath(__file__)) 
    # This assumes 'summe' is directly under project_root
    # e.g. d:\Workspace\Classes\TriTueNhanTao\vsumm-reinforce_re\summe\GT
    
    # If your 'summe' directory is located at 'd:\Workspace\Classes\TriTueNhanTao\summe',
    # you would need to adjust the base_path.
    # For example:
    # grand_parent_dir = os.path.dirname(os.path.dirname(project_root))
    # read_summe_gt_files(grand_parent_dir)

    # For now, assuming 'summe' is a subdirectory of where this script is.
    # If 'summe' is at 'd:\Workspace\Classes\TriTueNhanTao\vsumm-reinforce_re\summe'
    
    # Example 1: Read all files (previous behavior)
    # print("\nReading all files in summe/GT:")
    # all_contents = read_summe_gt_files()
    # if all_contents:
    #     print(f"\nSuccessfully processed {len(all_contents)} files.")
    # else:
    #     print("\nNo files were processed or directory was empty/not found.")

    # print("-" * 50)

    # Example 2: Read a specific .mat file
    # Replace 'Air_Force_One.mat' with an actual .mat filename in your summe/GT folder
    specific_mat_file = "Air_Force_One.mat" 
    print(f"\nReading specific file: {specific_mat_file}")
    single_file_content_mat = read_summe_gt_files(target_filename=specific_mat_file)
    if single_file_content_mat:
        print(f"\nSuccessfully processed {specific_mat_file}.")
        # Access content like: single_file_content_mat[specific_mat_file]
    else:
        print(f"\nCould not process {specific_mat_file} or it was not found.")

    print("-" * 50)

    # Example 3: Read a specific text file (if you have one)
    # Replace 'some_text_file.txt' with an actual text filename in your summe/GT folder
    # specific_text_file = "Exciting_Pole_Vault.txt" # Change this to a real text file name if available
    # print(f"\nReading specific file: {specific_text_file}")
    # single_file_content_text = read_summe_gt_files(target_filename=specific_text_file)
    # if single_file_content_text:
    #     print(f"\nSuccessfully processed {specific_text_file}.")
    #     # Access content like: single_file_content_text[specific_text_file]
    # else:
    #     print(f"\nCould not process {specific_text_file} or it was not found.")

    # To make it interactive, you could use:
    # chosen_file = input("Enter the filename to read (or press Enter to read all): ")
    # if chosen_file.strip() == "":
    #     contents = read_summe_gt_files()
    # else:
    #     contents = read_summe_gt_files(target_filename=chosen_file)
    # # ... then process contents ...
