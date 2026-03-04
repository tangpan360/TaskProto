import os
import json
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_eadro_root = os.path.dirname(os.path.dirname(_script_dir))
_project_root = os.path.dirname(os.path.dirname(_eadro_root))
input_path = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'label_gaia.csv')
output_path = os.path.join(_eadro_root, 'data')

def process_gaia_records():
    df = pd.read_csv(input_path)
    
    df['st_timestamp'] = pd.to_datetime(df['st_time']).astype(int) // 10**9
    df['ed_timestamp'] = df['st_timestamp'] + 600
    
    start_time = int(df['st_timestamp'].min())
    end_time = int(df['ed_timestamp'].max())
    
    faults = []
    for _, row in df.iterrows():
        faults.append({
            'instance': row['instance'],
            'fault_type': row['anomaly_type'],
            's': int(row['st_timestamp']),
            'e': int(row['ed_timestamp'])
        })
    
    processed_records = {
        'start': start_time,
        'end': end_time,
        'faults': faults
    }
    
    save_path = os.path.join(output_path, 'parsed_data', 'GAIA', 'records0.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(processed_records, f)


if __name__ == "__main__":
    process_gaia_records()