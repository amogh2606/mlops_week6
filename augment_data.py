import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Define the source files that will be combined to create the augmented version
SOURCE_FILES = [
    "data/V0_RAW.csv",
    "data/V1_data.csv",
    "data/V2_data.csv"
]

OUTPUT_FILE = "data/V3_augmented.csv"
AUGMENTATION_VERSION_NAME = "V3_augmented_data"

def augment_iris_data():
    """
    Loads V0, V1, and V2 data sets, concatenates them to simulate
    a comprehensive data addition, and saves the new V3 version locally.
    """
    all_dfs = []

    # 1. Load the existing data versions
    for file_path in SOURCE_FILES:
        if not os.path.exists(file_path):
            print(f"Error: Required source file not found at {file_path}. Ensure all V0, V1, and V2 data files have been copied locally.")
            return False
        
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            print(f"Loaded {len(df)} rows from {file_path}.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False

    if not all_dfs:
        print("No data frames loaded. Aborting augmentation.")
        return False

    # 2. Combine the data frames (Augmentation by Concatenation)
    df_augmented = pd.concat(all_dfs, ignore_index=True)

    # 3. Save the new augmented version
    df_augmented.to_csv(OUTPUT_FILE, index=False)
    
    # 4. Report statistics
    original_size = sum(len(df) for df in all_dfs)
    new_size = len(df_augmented)
    
    print(f"\n--- Data Augmentation Complete ---")
    print(f"Total rows from combined sources (V0+V1+V2): {original_size} rows")
    print(f"New V3 data file size: {new_size} rows")
    print(f"New data file saved locally: {OUTPUT_FILE}")
    
    return True

if __name__ == "__main__":
    if augment_iris_data():
        print(f"\nNext Steps: Add '{OUTPUT_FILE}' to DVC using 'dvc add', commit the metadata, and run training on it.")
