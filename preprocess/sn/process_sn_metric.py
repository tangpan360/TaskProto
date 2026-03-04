import os
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))

def process_sn_metrics():
    # 1) Path configuration
    raw_data_dir = os.path.join(_project_root, "data", "raw_data", "sn", "data")
    label_path = os.path.join(_project_root, "data", "processed_data", "sn", "label_sn.csv")
    output_dir = os.path.join(_project_root, "data", "processed_data", "sn", "metric")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2) Read labels to get the service list
    label_df = pd.read_csv(label_path)
    
    # 3) List all experiment folders
    exp_folders = sorted([f for f in os.listdir(raw_data_dir) if f.startswith("SN.") and os.path.isdir(os.path.join(raw_data_dir, f))])
    
    all_services = sorted(label_df['service'].unique())
    print(f"Processing metric data for {len(all_services)} services...")
    
    # 4) Process each service
    for service in tqdm(all_services):
        service_dfs = []
        
        # Scan all experiment folders to collect this service's metric data
        for folder in exp_folders:
            csv_path = os.path.join(raw_data_dir, folder, "metrics", f"{service}.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Ensure timestamps are integer seconds
                df['timestamp'] = df['timestamp'].astype(int)
                service_dfs.append(df)
        
        if service_dfs:
            # Merge, sort by timestamp, and drop duplicates
            full_df = pd.concat(service_dfs).sort_values('timestamp').reset_index(drop=True)
            full_df = full_df.drop_duplicates(subset=['timestamp'])
            
            # Save the processed metric file
            save_path = os.path.join(output_dir, f"{service}_metric.csv")
            full_df.to_csv(save_path, index=False)
        else:
            print(f"❌ Error: no metric file found for service {service}")

if __name__ == "__main__":
    process_sn_metrics()
