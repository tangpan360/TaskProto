#!/usr/bin/env python3
"""
Log processing utility - extract normal/anomalous data and analyze patterns.
Anomaly window definition: a fixed 600-second window starting from the label start time.
"""

import os
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

# Drain-related imports
import sys
# Get project root directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)
from utils.drain.drain_template_extractor import init_drain, extract_templates


def load_anomaly_periods(label_file_path):
    """
    Load anomaly time windows (fixed 600-second windows).
    
    Args:
        label_file_path (str): Path to the label file.
        
    Returns:
        list: List of (start_ms, end_ms, data_type) tuples.
    """
    print("Loading anomaly periods...")
    
    # Read label file
    label_df = pd.read_csv(label_file_path)
    
    # Convert time to timestamps; each anomaly window lasts 600 seconds from start.
    anomaly_periods = []
    for _, row in label_df.iterrows():
        # Convert start time string to timestamp (ms).
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000  # start + 600 seconds
        data_type = row.get('data_type', 'unknown')  # default: unknown
        anomaly_periods.append((st_time, ed_time, data_type))
    
    print(f"Loaded {len(anomaly_periods)} anomaly periods")
    return anomaly_periods


# def extract_anomaly_log_data(log_dir, anomaly_periods, output_dir):
#     """
#     Extract log rows within anomaly windows from all files under MicroSS/business.
#     Save to preprocess/logs using the original filenames.
    
#     Args:
#         log_dir (str): Source log directory
#         anomaly_periods (list): Anomaly periods
#         output_dir (str): Output directory
#     """
#     print("=== Start extracting anomaly log data ===")
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # List log files
#     log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
#     print(f"Found {len(log_files)} log files: {log_files}")
    
#     # Process each file
#     for log_file in log_files:
#         print(f"\nProcessing file: {log_file}")
#         log_file_path = os.path.join(log_dir, log_file)
        
#         # Read log file
#         log_df = pd.read_csv(log_file_path)
#         original_count = len(log_df)
#         print(f"  Rows read: {original_count:,}")
        
#         # Extract timestamp from the message field and convert to timestamp (ms)
#         # Assumed message format: "2021-07-01 10:54:22,639 | ..."
#         log_df['timestamp'] = log_df['message'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
#         log_df['timestamp_ts'] = pd.to_datetime(log_df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f').astype('int64') // 10**6
        
#         # Drop rows with unparseable timestamps
#         log_df = log_df.dropna(subset=['timestamp_ts'])
        
#         # Create a mask for normal periods (then invert it to get anomaly rows)
#         normal_mask = create_selection_mask(log_df['timestamp_ts'], anomaly_periods, data_type_filter=None)
        
#         # Invert mask to extract anomaly rows
#         anomaly_data = log_df[~normal_mask].copy()
#         anomaly_count = len(anomaly_data)
        
#         # Save to the same filename
#         output_file_path = os.path.join(output_dir, log_file)
#         anomaly_data.to_csv(output_file_path, index=False)
        
#         print(f"  Extracted anomaly rows: {anomaly_count:,} ({anomaly_count/original_count:.2%})")
#         print(f"  Saved to: {output_file_path}")


def process_single_log_file(args):
    """
    Worker function for processing a single log file (for multiprocessing).
    
    Args:
        args (tuple): (log_file, log_dir, anomaly_periods, output_dir)
        
    Returns:
        dict: Processing stats/result
    """
    log_file, log_dir, anomaly_periods, output_dir = args
    
    try:
        print(f"\n[Process] Processing file: {log_file}")
        log_file_path = os.path.join(log_dir, log_file)
        
        # Read log file
        log_df = pd.read_csv(log_file_path)
        original_count = len(log_df)
        print(f"  [Process] Rows read: {original_count:,}")
        
        # Extract timestamp from message and convert to timestamp (ms).
        # Assumed message format: "2021-07-01 10:54:22,639 | ..."
        log_df['timestamp'] = log_df['message'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')
        log_df['timestamp_ts'] = pd.to_datetime(log_df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f').astype('int64') // 10**6
        
        # Drop rows with unparseable timestamps
        log_df = log_df.dropna(subset=['timestamp_ts'])
        
        # Create a mask for anomaly windows
        anomaly_mask = create_selection_mask(log_df['timestamp_ts'], anomaly_periods)
        
        # Extract rows within anomaly windows
        anomaly_data = log_df[anomaly_mask]
        anomaly_count = len(anomaly_data)
        
        # Extract instance name from filename (3rd segment)
        splits = log_file.replace('.csv', '').split('_')
        instance_name = splits[2]  # 3rd segment as instance name
        
        # Save to renamed file
        output_filename = f"{instance_name}_log.csv"
        output_file_path = os.path.join(output_dir, output_filename)
        anomaly_data.to_csv(output_file_path, index=False)
        
        print(f"  [Process] Extracted anomaly rows: {anomaly_count:,} ({anomaly_count/original_count:.2%})")
        print(f"  [Process] Saved to: {output_file_path}")
        
        return {
            'file': log_file,
            'original_count': original_count,
            'anomaly_count': anomaly_count,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"  [Process] Error processing {log_file}: {e}")
        return {
            'file': log_file,
            'original_count': 0,
            'anomaly_count': 0,
            'status': 'error',
            'error': str(e)
        }


def extract_anomaly_log_data_multiprocess(log_dir, anomaly_periods, output_dir, n_processes=None):
    """
    Extract log rows within anomaly windows from all files under MicroSS/business using multiprocessing.
    Save to preprocess/logs with per-instance renamed filenames.
    
    Args:
        log_dir (str): Source log directory.
        anomaly_periods (list): Anomaly periods.
        output_dir (str): Output directory.
        n_processes (int, optional): Number of processes (default: CPU cores, capped by file count).
    """
    print("=== Start extracting anomaly log data (multiprocessing) ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List all log files
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    print(f"Found {len(log_files)} log files: {log_files}")
    
    # Determine number of processes
    if n_processes is None:
        n_processes = min(cpu_count(), len(log_files))  # do not exceed file count
    
    print(f"Using {n_processes} processes")
    
    # Prepare argument list
    args_list = [(log_file, log_dir, anomaly_periods, output_dir) for log_file in log_files]
    
    # Run multiprocessing
    start_time = datetime.now()
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_log_file, args_list)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Aggregate results
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    total_original = sum(r['original_count'] for r in results if r['status'] == 'success')
    total_anomaly = sum(r['anomaly_count'] for r in results if r['status'] == 'success')
    
    print("\n=== Multiprocessing finished ===")
    print(f"Elapsed: {processing_time:.2f} seconds")
    print(f"Succeeded: {success_count} files")
    print(f"Failed: {error_count} files")
    print(f"Total original rows: {total_original:,}")
    print(f"Total anomaly rows: {total_anomaly:,} ({total_anomaly/total_original:.2%} anomaly rate)")
    
    # Show failed files
    if error_count > 0:
        print("\nFailed files:")
        for r in results:
            if r['status'] == 'error':
                print(f"  {r['file']}: {r['error']}")
    
    return results


def extract_service_durations_by_timesegments(trace_dir, anomaly_periods, output_file):
    """
    Read all trace files and bucket data into 20 x 30-second segments for each 600-second anomaly window.
    For each segment, collect all duration values per service and save as JSON.
    
    Args:
        trace_dir (str): Trace directory.
        anomaly_periods (list): Anomaly periods; each element is a tuple (start_ms, end_ms, ...).
        output_file (str): Output JSON file path.
    """
    import json
    
    print("=== Start extracting per-service duration series ===")
    
    # 1) List all trace files and load them into memory
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    print(f"Found {len(trace_files)} trace files")
    
    # 2) Read all files into memory
    print("Loading all trace files into memory...")
    trace_dataframes = {}  # {instance_name: dataframe}
    
    from tqdm import tqdm
    for trace_file in tqdm(trace_files, desc="Reading files", unit="file"):
        # Extract instance name
        parts = trace_file.split('_')
        if len(parts) < 3:
            continue
        instance_name = parts[2]  # e.g., dbservice1
        
        try:
            trace_file_path = os.path.join(trace_dir, trace_file)
            df = pd.read_csv(trace_file_path)
            trace_dataframes[instance_name] = df
            print(f"  Loaded {instance_name}: {len(df):,} rows")
        except Exception as e:
            print(f"  Warning: failed to read {trace_file}: {e}")
            continue
    
    print(f"Loaded {len(trace_dataframes)} files into memory")
    
    # 3) Initialize result dict
    result_data = {}
    
    print(f"\nProcessing {len(anomaly_periods)} anomaly windows...")
    
    # 4) Process each anomaly window
    for period_idx, (start_time, end_time) in enumerate(tqdm(anomaly_periods, desc="Processing anomaly windows", unit="period")):
        # Split 600 seconds into 20 segments of 30 seconds.
        for segment in range(20):
            segment_start = int(start_time + segment * 30 * 1000)  # timestamp in ms
            segment_end = int(segment_start + 30 * 1000)
            
            segment_key = str(segment_start)
            if segment_key not in result_data:
                result_data[segment_key] = {}
            
            # 5) Process each instance's in-memory data
            for instance_name, df in trace_dataframes.items():
                try:
                    # Filter rows within this segment
                    mask = (df['start_time_ts'] >= segment_start) & (df['start_time_ts'] < segment_end)
                    segment_data = df[mask]
                    
                    # If this segment has data, append to results
                    if len(segment_data) > 0:
                        durations = segment_data['duration'].tolist()
                        
                        # Use full instance name as service key (e.g., dbservice1, dbservice2)
                        if instance_name not in result_data[segment_key]:
                            result_data[segment_key][instance_name] = []
                        
                        result_data[segment_key][instance_name].extend(durations)
                
                except Exception as e:
                    print(f"    Warning: error processing {instance_name} at segment {segment_key}: {e}")
                    continue
    
    # 5) Remove empty segments
    result_data = {k: v for k, v in result_data.items() if v}
    
    # 6) Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # 7) Summary stats
    total_segments = len(result_data)
    total_services = set()
    total_values = 0
    
    for segment_data in result_data.values():
        total_services.update(segment_data.keys())
        for service_values in segment_data.values():
            total_values += len(service_values)
    
    print("\n=== Extraction finished ===")
    print(f"Anomaly windows: {len(anomaly_periods)}")
    print(f"Valid segments: {total_segments}")
    print(f"Services: {len(total_services)} ({list(total_services)})")
    print(f"Total data points: {total_values:,}")
    print(f"Avg per segment: {total_values/total_segments:.1f} data points")
    print(f"Output file: {output_file}")
    
    return result_data


def create_selection_mask(times, target_periods):
    """
    Create a boolean selection mask for target time windows.
    
    Args:
        times (pd.Series): Timestamp series.
        target_periods (list): List of periods, each element is (start_time, end_time, ...).
        
    Returns:
        pd.Series: Boolean mask. True means inside a target window; False otherwise.
    """
    # Initialize as all False.
    is_in_target = pd.Series(False, index=times.index)
    
    # Mark timestamps that fall into any target period as True.
    for start_time, end_time, _ in target_periods:
        # Find timestamps within the current window.
        in_current_period = (times >= start_time) & (times <= end_time)
        
        # Mark as selected.
        is_in_target.loc[in_current_period] = True
    
    return is_in_target


def train_drain_from_logs(log_dir, output_dir, anomaly_periods=None):
    """
    Train a Drain template miner from log data, using only data_type='train' anomaly windows.

    Args:
        log_dir (str): Source log directory.
        output_dir (str): Output directory.
        anomaly_periods (list, optional): List of anomaly periods (triples). If None, use all data.

    Returns:
        TemplateMiner: Trained Drain model.
    """
    print("=== Start training Drain templates from logs ===")
    print("Using only anomaly windows with data_type='train' for training")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Filter training windows
    train_anomaly_periods = [(st, ed, dt) for st, ed, dt in anomaly_periods if dt == 'train']
    print(f"Selected {len(train_anomaly_periods)} training windows from {len(anomaly_periods)} total anomaly windows")

    # List log files
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]

    # Collect log messages
    all_log_messages = []
    total_logs = 0

    for log_file in tqdm(log_files, desc="Reading log files"):
        log_file_path = os.path.join(log_dir, log_file)

        try:
            # Read log file
            log_df = pd.read_csv(log_file_path)
            print(f"\nProcessing {log_file}: {len(log_df):,} rows")

            # Drop rows without parseable timestamps
            log_df = log_df.dropna(subset=['timestamp_ts'])
            print(f"  Rows with valid timestamps: {len(log_df):,}")

            # Build a mask for training windows
            train_mask = create_selection_mask(log_df['timestamp_ts'], train_anomaly_periods)
            filtered_df = log_df[train_mask]  # use mask directly to select training-window rows
            print(f"  Training rows after filtering: {len(filtered_df):,}")

            # Extract message strings
            messages = filtered_df['message'].tolist()
            all_log_messages.extend(messages)
            total_logs += len(messages)

        except Exception as e:
            print(f"  Warning: error processing {log_file}: {e}")
            continue

    print(f"\nCollected {total_logs:,} log messages for training")

    # Train Drain templates
    print("\nTraining Drain templates...")
    drain_model_path = os.path.join(output_dir, "gaia_drain.pkl")
    
    # Gaia Drain config path
    gaia_config_path = os.path.join(_project_root, "preprocess", "gaia", "gaia_drain3.ini")

    miner = extract_templates(
        log_list=all_log_messages,
        save_pth=drain_model_path,
        config_path=gaia_config_path
    )

    # Save template info
    template_csv_path = os.path.join(output_dir, "gaia_templates.csv")
    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)

    template_data = {
        'template_id': [],
        'template': [],
        'count': [],
        'percentage': []
    }

    for cluster in sorted_clusters:
        template_data['template_id'].append(cluster.cluster_id)
        template_data['template'].append(cluster.get_template())
        template_data['count'].append(cluster.size)
        template_data['percentage'].append(cluster.size / total_logs * 100)

    template_df = pd.DataFrame(template_data)
    template_df.to_csv(template_csv_path, index=False)

    return miner


def load_drain_model(model_path):
    """
    Load a trained Drain model.
    """
    with open(model_path, 'rb') as f:
        miner = pickle.load(f)
    return miner

def add_template_columns_single_file(args):
    """
    Add `template` and `template_id` columns for a single CSV file.

    Args:
        args (tuple): (file_path, model_path)

    Returns:
        str: Result message
    """
    file_path, model_path = args
    miner = load_drain_model(model_path)

    try:
        df = pd.read_csv(file_path)

        if 'template' in df.columns and 'template_id' in df.columns:
            print(f"File {os.path.basename(file_path)} already has template/template_id columns; skipping")
            return f"Skipped {os.path.basename(file_path)}"

        templates = []
        template_ids = []
        for message in tqdm(df['message'], desc=f"Processing {os.path.basename(file_path)}"):
            match = miner.match(message)
            if match:
                template = match.get_template()
                template_id = match.cluster_id
            else:
                template = "Unseen"
                template_id = -1  # unseen logs use -1 as ID
            templates.append(template)
            template_ids.append(template_id)
        
        df['template'] = templates
        df['template_id'] = template_ids

        df.to_csv(file_path, index=False)

        return f"Done {os.path.basename(file_path)}: processed {len(df)} rows"
    
    except Exception as e:
        return f"Error processing {os.path.basename(file_path)}: {str(e)}"


def add_template_columns_multiprocess(logs_dir, model_path, numprocess=None):
    """
    Add `template` and `template_id` columns for all CSV files under logs_dir.

    Args:
        logs_dir (str): Log directory path.
        model_path (str): Trained Drain model path.
        num_processes (int): Number of processes (default: CPU cores).
    """
    csv_files = [f for f in os.listdir(logs_dir) if f.endswith('.csv')]
    file_paths = [os.path.join(logs_dir, f) for f in csv_files]
    
    if numprocess is None:
        num_processes = min(cpu_count(), len(file_paths))
    
    print(f"Using {num_processes} processes")

    args_list = [(file_path, model_path) for file_path in file_paths]

    with Pool(processes=num_processes) as pool:
        results = pool.map(add_template_columns_single_file, args_list)

    print("\nResults:")
    for result in results:
        print(result)


def main():
    """
    Entry point.
    """
    # Define paths
    label_file = os.path.join(_project_root, "data", "processed_data", "gaia", "label_gaia.csv")
    log_dir = os.path.join(_project_root, "data", "raw_data", "gaia", "business")
    output_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "log")
    drain_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "drain_models")    
    
    # 1) Load anomaly periods
    anomaly_periods = load_anomaly_periods(label_file)

    # Extract anomaly log data (multiprocessing)
    extract_anomaly_log_data_multiprocess(log_dir, anomaly_periods, output_dir)
    
    # Train Drain templates (using training windows)
    train_drain_from_logs(output_dir, drain_dir, anomaly_periods)

    # Add template columns
    model_path = os.path.join(drain_dir, "gaia_drain.pkl")
    add_template_columns_multiprocess(output_dir, model_path)
    

if __name__ == "__main__":
    main()
