import os
import shutil

# Get the directory of this script.
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory.
_project_root = os.path.dirname(os.path.dirname(_script_dir))

def process_gaia_label():
    """
    Copy `label_gaia.csv` from `raw_data` to `processed_data`.
    """
    # Source file path
    source_path = os.path.join(_project_root, 'data', 'raw_data', 'gaia', 'label_gaia.csv')
    
    # Output directory
    output_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia')
    
    # Target file path
    target_path = os.path.join(output_dir, 'label_gaia.csv')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
        
    # Copy file
    shutil.copy2(source_path, target_path)

    print(f"Successfully copied label file to: {target_path}")

if __name__ == "__main__":
    process_gaia_label()
