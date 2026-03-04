import os
import pandas as pd
from tqdm import tqdm

from util import Info

_script_dir = os.path.dirname(os.path.abspath(__file__))
_eadro_root = os.path.dirname(os.path.dirname(_script_dir))
_project_root = os.path.dirname(os.path.dirname(_eadro_root))
input_path = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'log')
output_path = os.path.join(_eadro_root, 'data')

def parse_gaia_logs():
    log_files = [f for f in os.listdir(input_path) if f.endswith('_log.csv')]
    
    all_logs = {'timestamp': [], 'service': [], 'events': []}
    
    for log_file in tqdm(log_files, desc="Processing log files"):
        df = pd.read_csv(os.path.join(input_path, log_file))
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {log_file}", leave=False):
            timestamp = float(row['timestamp_ts'] / 1000)
            service = row['service']
            template = row['template']
            
            all_logs['timestamp'].append(timestamp)
            all_logs['service'].append(service)
            all_logs['events'].append(template)
    
    result_df = pd.DataFrame(all_logs)
    save_path = os.path.join(output_path, 'parsed_data/GAIA/logs0.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    parse_gaia_logs()
