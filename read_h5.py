import h5py
import os

def print_h5_structure(item, indent=''):
    """Recursively prints the structure of an HDF5 file."""
    if isinstance(item, h5py.File):
        print(f"{indent}File: {item.filename}")
        for key in item.keys():
            print_h5_structure(item[key], indent + "  ")
    elif isinstance(item, h5py.Group):
        print(f"{indent}Group: {item.name}")
        for key in item.keys():
            print_h5_structure(item[key], indent + "  ")
    elif isinstance(item, h5py.Dataset):
        print(f"{indent}Dataset: {item.name} (Shape: {item.shape}, Dtype: {item.dtype})")
        # Optionally, print some data if the dataset is small
        # if item.size < 10:
        #     print(f"{indent}  Data: {item[...]}")
    else:
        print(f"{indent}Unknown item: {item.name}")

def extract_data_recursive(item, data_dict, path=''):
    """Recursively extracts datasets into a dictionary."""
    if isinstance(item, h5py.Group) or isinstance(item, h5py.File):
        for key in item.keys():
            current_path = f"{path}/{key}" if path else key
            extract_data_recursive(item[key], data_dict, current_path)
    elif isinstance(item, h5py.Dataset):
        data_dict[path] = item[...]

def read_h5_file(file_path):
    """Reads and prints the structure of an HDF5 file, and extracts all data."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None

    extracted_data = {}
    try:
        with h5py.File(file_path, 'r') as hf:
            print(f"Successfully opened HDF5 file: {file_path}")
            print("HDF5 File Structure:")
            print_h5_structure(hf)
            
            print("\nExtracting data...")
            extract_data_recursive(hf, extracted_data)
            print("Data extraction complete.")
            return extracted_data
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None

if __name__ == '__main__':
    # Define the path to the HDF5 file
    # Assuming 'result' is a subdirectory in the same directory as this script
    # or provide an absolute path.
    h5_file_path = os.path.join(os.path.dirname(__file__), 'result', 'Air_Force_One.h5')
    
    # For a fixed path as requested:
    # h5_file_path = 'd:/Workspace/Classes/TriTueNhanTao/vsumm-reinforce_re/result/test.h5'

    print(f"Attempting to read and extract data from: {h5_file_path}")
    all_data = read_h5_file(h5_file_path)

    if all_data:
        print("\nExtracted Data (Dataset Path: Shape, Dtype):")
        for path, data in all_data.items():
            if hasattr(data, 'shape') and hasattr(data, 'dtype'):
                print(f"  Dataset Path: {path}")
                print(f"    Shape: {data.shape}, Dtype: {data.dtype}")
                print(f"    Content:\n{data}\n")
            else:
                # For attributes or other non-dataset items if any were captured
                print(f"  Item Path: {path}")
                print(f"    Type: {type(data)}")
                print(f"    Content:\n{data}\n")
        
        # Example: Print content of a specific dataset if it exists
        # target_dataset_path = 'video_1/features' # Change this to an actual dataset path from your file
        # if target_dataset_path in all_data:
        #     print(f"\nContent of '{target_dataset_path}':")
        #     print(all_data[target_dataset_path])
        # else:
        #     print(f"\nDataset '{target_dataset_path}' not found in extracted data.")

    # Example of how to access a specific dataset if you know its path
    # try:
    #     with h5py.File(h5_file_path, 'r') as hf:
    #         # Replace 'video_1/features' with the actual path to your dataset
    #         if 'video_1/features' in hf:
    #             data = hf['video_1/features'][:]
    #             print(f"\nData from 'video_1/features':\n{data}")
    #         else:
    #             print(f"\nDataset 'video_1/features' not found in {h5_file_path}")
    # except Exception as e:
    #     print(f"Error accessing specific dataset: {e}")
