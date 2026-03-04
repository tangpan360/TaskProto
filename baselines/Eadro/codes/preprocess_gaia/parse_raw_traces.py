import os
import json
import pandas as pd
from tqdm import tqdm

from util import Info

_script_dir = os.path.dirname(os.path.abspath(__file__))
_eadro_root = os.path.dirname(os.path.dirname(_script_dir))
_project_root = os.path.dirname(os.path.dirname(_eadro_root))
input_path = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'trace')
output_path = os.path.join(_eadro_root, 'data')

def parse_gaia_traces():
    trace_files = [f for f in os.listdir(input_path) if f.endswith('_trace.csv')]
    
    t_s_l = []
    for trace_file in tqdm(trace_files, desc="Processing trace files"):
        df = pd.read_csv(os.path.join(input_path, trace_file))
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {trace_file}", leave=False):
            timestamp = int(row['start_time_ts'] / 1000)
            service_name = row['service_name']
            latency = float(row['duration'] / 1000)
            t_s_l.append((timestamp, service_name, latency))
        
    unique_timestamps = sorted(list(set([t for t, _, _ in t_s_l])))
    processed_data = {str(t): {} for t in unique_timestamps}
    for t, s, l in tqdm(t_s_l, desc="Building processed data"):
        processed_data[str(t)][s] = processed_data[str(t)].get(s, []) + [l]
        
    save_path = os.path.join(output_path, 'parsed_data/GAIA/traces0.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(processed_data, f)

if __name__ == '__main__':
    parse_gaia_traces()
