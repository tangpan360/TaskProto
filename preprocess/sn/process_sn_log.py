import os
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
from datetime import datetime

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)

# Import the unified Drain module
from utils.drain.drain_template_extractor import extract_templates

def parse_sn_log_timestamp(log_str):
    """
    Parse SN log timestamp.

    Format: [2022-Apr-17 10:12:50.490796]
    """
    pattern = r"\[(\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]"
    match = re.search(pattern, log_str)
    if not match:
        return None
    
    time_str = match.group(1)
    try:
        # Parse time string (includes English month abbreviation).
        dt = pd.to_datetime(time_str, format="%Y-%b-%d %H:%M:%S.%f", utc=True)
        ts = dt.timestamp()
        return ts
    except Exception as e:
        return None

def process_sn_logs():
    print("=== Start processing SN log data (custom Drain config) ===")
    
    # 1) Configure paths
    raw_data_dir = os.path.join(_project_root, "data", "raw_data", "sn", "data")
    label_path = os.path.join(_project_root, "data", "processed_data", "sn", "label_sn.csv")
    output_dir = os.path.join(_project_root, "data", "processed_data", "sn", "log")
    drain_model_dir = os.path.join(_project_root, "data", "processed_data", "sn", "drain_models")
    
    # Custom Drain config file
    drain_config_path = os.path.join(_script_dir, "sn_drain3.ini")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(drain_model_dir, exist_ok=True)
    
    # 2) Load labels to identify training time windows
    print("Loading label data...")
    label_df = pd.read_csv(label_path)
    
    # Identify training intervals
    train_samples = label_df[label_df['data_type'] == 'train']
    train_intervals = []
    for _, row in train_samples.iterrows():
        st = pd.to_datetime(row['st_time'], utc=True).timestamp()
        ed = st + row['duration']
        train_intervals.append((st, ed))
    
    # 3) Collect all logs
    exp_folders = sorted([f for f in os.listdir(raw_data_dir) if f.startswith("SN.") and os.path.isdir(os.path.join(raw_data_dir, f))])
    all_logs = [] 
    
    print(f"Parsing logs from {len(exp_folders)} experiment folders...")
    for folder in tqdm(exp_folders, desc="Parsing log files"):
        log_json_path = os.path.join(raw_data_dir, folder, "logs.json")
        if not os.path.exists(log_json_path):
            continue
            
        with open(log_json_path, 'r') as f:
            try:
                logs_dict = json.load(f)
            except json.JSONDecodeError:
                continue
            
        for raw_service, log_list in logs_dict.items():
            service = raw_service
            
            for log_msg in log_list:
                ts = parse_sn_log_timestamp(log_msg)
                if ts is not None:
                    all_logs.append({
                        'timestamp': ts,
                        'service': service,
                        'message': log_msg
                    })
    
    print(f"Collected {len(all_logs)} valid log messages.")
    logs_df = pd.DataFrame(all_logs)
    
    # 4) Filter training logs
    print("Filtering training logs...")
    logs_df = logs_df.sort_values('timestamp')
    train_intervals.sort()
    merged_intervals = []
    if train_intervals:
        curr_start, curr_end = train_intervals[0]
        for next_start, next_end in train_intervals[1:]:
            if next_start <= curr_end:
                curr_end = max(curr_end, next_end)
            else:
                merged_intervals.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_intervals.append((curr_start, curr_end))
    
    is_train = pd.Series(False, index=logs_df.index)
    for start, end in merged_intervals:
        # Mark timestamps in the current training interval.
        in_current_period = (logs_df['timestamp'] >= start) & (logs_df['timestamp'] < end)
        # Mark as training (True)
        is_train.loc[in_current_period] = True
        
    train_logs_df = logs_df[is_train]
    train_messages = train_logs_df['message'].tolist()
    print(f"Selected {len(train_messages)} training log messages for Drain training")
    
    if not train_messages:
        print("⚠️  Error: No training logs found. Check timestamp alignment.")
        return

    # 5) Train Drain (with custom config)
    drain_model_path = os.path.join(drain_model_dir, "sn_drain.pkl")
    
    miner = extract_templates(
        log_list=train_messages,
        save_pth=drain_model_path,
        config_path=drain_config_path
    )
    
    # Save template statistics
    template_csv_path = os.path.join(drain_model_dir, "sn_templates.csv")
    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)
    template_data = {
        'template_id': [c.cluster_id for c in sorted_clusters],
        'template': [c.get_template() for c in sorted_clusters],
        'count': [c.size for c in sorted_clusters]
    }
    pd.DataFrame(template_data).to_csv(template_csv_path, index=False)
    
    # 6) Match templates for all logs
    print("Matching templates...")
    all_templates = []
    all_template_ids = []
    
    for msg in tqdm(logs_df['message'], desc="Matching"):
        match = miner.match(msg)
        if match:
            all_templates.append(match.get_template())
            all_template_ids.append(match.cluster_id)
        else:
            all_templates.append("Unseen")
            all_template_ids.append(-1)
            
    logs_df['template'] = all_templates
    logs_df['template_id'] = all_template_ids
    
    # 7) Save per-service CSVs
    unique_services = logs_df['service'].unique()
    for service in tqdm(unique_services, desc="Saving CSVs"):
        service_logs = logs_df[logs_df['service'] == service].copy().sort_values('timestamp')
        save_path = os.path.join(output_dir, f"{service}_log.csv")
        service_logs.to_csv(save_path, index=False)
        
    print("=== Done ===")

if __name__ == "__main__":
    process_sn_logs()
