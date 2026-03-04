#!/usr/bin/env python3
"""
Trace processing utility - extract normal/anomalous data and analyze patterns.
Anomaly window definition: a fixed 600-second window starting from the label start time.
"""

import os
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
import sys
import os
# Get project root directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)
warnings.filterwarnings('ignore')


def load_anomaly_periods(label_file_path):
    """
    Load anomaly time windows (fixed 600-second windows).
    
    Args:
        label_file_path (str): Path to the label file.
        
    Returns:
        list: List of (start_ms, end_ms, data_type) tuples.
    """
    print("Loading anomaly periods...")
    
    # Read label file.
    label_df = pd.read_csv(label_file_path)
    
    # Convert to timestamps; each anomaly window lasts 600 seconds from start.
    anomaly_periods = []
    for _, row in label_df.iterrows():
        # Convert start time string to timestamp (ms).
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000  # start + 600 seconds
        data_type = row.get('data_type', 'unknown')  # default: unknown
        anomaly_periods.append((st_time, ed_time, data_type))
    
    print(f"Loaded {len(anomaly_periods)} anomaly periods")
    return anomaly_periods


def extract_anomaly_trace_data(trace_dir, anomaly_periods, output_dir):
    """
    Extract trace rows within anomaly windows from all files under MicroSS/trace.
    Save to preprocess/trace using per-instance renamed filenames.
    
    Args:
        trace_dir (str): Source trace directory.
        anomaly_periods (list): Anomaly periods.
        output_dir (str): Output directory.
    """
    print("=== Start extracting anomaly trace data ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List trace files
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    print(f"Found {len(trace_files)} trace files: {trace_files}")
    
    # Process each file
    for trace_file in trace_files:
        print(f"\nProcessing file: {trace_file}")
        trace_file_path = os.path.join(trace_dir, trace_file)
        
        # Read trace file
        trace_df = pd.read_csv(trace_file_path)
        original_count = len(trace_df)
        print(f"  Rows read: {original_count:,}")
        
        # Convert start_time/end_time to timestamps (ms).
        trace_df['start_time_ts'] = pd.to_datetime(trace_df['start_time'], format='mixed').astype('int64') // 10**6
        trace_df['end_time_ts'] = pd.to_datetime(trace_df['end_time'], format='mixed').astype('int64') // 10**6
        
        # Build anomaly window mask (and normal mask as its inverse).
        anomaly_mask = create_selection_mask(trace_df['start_time_ts'], anomaly_periods)
        normal_mask = ~anomaly_mask
        
        # Extract anomaly rows.
        anomaly_data = trace_df[anomaly_mask].copy()
        anomaly_count = len(anomaly_data)
        
        # Compute duration (ms).
        anomaly_data['duration'] = anomaly_data['end_time_ts'] - anomaly_data['start_time_ts']
        
        # Extract instance name from filename (3rd segment).
        splits = trace_file.replace('.csv', '').split('_')
        instance_name = splits[2]  # 3rd segment as instance name
        
        # Save to renamed file
        output_filename = f"{instance_name}_trace.csv"
        output_file_path = os.path.join(output_dir, output_filename)
        anomaly_data.to_csv(output_file_path, index=False)
        
        print(f"  Extracted anomaly rows: {anomaly_count:,} ({anomaly_count/original_count:.2%})")
        print(f"  Saved to: {output_file_path}")


def process_single_trace_file(args):
    """
    Worker function for processing a single trace file (for multiprocessing).
    
    Args:
        args (tuple): (trace_file, trace_dir, anomaly_periods, output_dir)
        
    Returns:
        dict: Processing stats/result
    """
    trace_file, trace_dir, anomaly_periods, output_dir = args
    
    try:
        print(f"\n[Process] Processing file: {trace_file}")
        trace_file_path = os.path.join(trace_dir, trace_file)
        
        # Read trace file
        trace_df = pd.read_csv(trace_file_path)
        original_count = len(trace_df)
        print(f"  [Process] Rows read: {original_count:,}")
        
        # Convert start_time/end_time to timestamps (ms).
        trace_df['start_time_ts'] = pd.to_datetime(trace_df['start_time'], format='mixed').astype('int64') // 10**6
        trace_df['end_time_ts'] = pd.to_datetime(trace_df['end_time'], format='mixed').astype('int64') // 10**6
        
        # Build anomaly window mask.
        anomaly_mask = create_selection_mask(trace_df['start_time_ts'], anomaly_periods)
        
        # Extract anomaly rows.
        anomaly_data = trace_df[anomaly_mask].copy()
        anomaly_count = len(anomaly_data)
        
        # Compute duration (ms).
        anomaly_data['duration'] = anomaly_data['end_time_ts'] - anomaly_data['start_time_ts']
        
        # Extract instance name from filename (3rd segment).
        splits = trace_file.replace('.csv', '').split('_')
        instance_name = splits[2]  # 3rd segment as instance name
        
        # Save to renamed file
        output_filename = f"{instance_name}_trace.csv"
        output_file_path = os.path.join(output_dir, output_filename)
        anomaly_data.to_csv(output_file_path, index=False)
        
        print(f"  [Process] Extracted anomaly rows: {anomaly_count:,} ({anomaly_count/original_count:.2%})")
        print(f"  [Process] Saved to: {output_file_path}")
        
        return {
            'file': trace_file,
            'original_count': original_count,
            'anomaly_count': anomaly_count,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"  [Process] Error processing {trace_file}: {e}")
        return {
            'file': trace_file,
            'original_count': 0,
            'anomaly_count': 0,
            'status': 'error',
            'error': str(e)
        }


def extract_anomaly_trace_data_multiprocess(trace_dir, anomaly_periods, output_dir, n_processes=None):
    """
    Extract trace rows within anomaly windows from all files under MicroSS/trace using multiprocessing.
    Save to preprocess/trace using per-instance renamed filenames.
    
    Args:
        trace_dir (str): Source trace directory.
        anomaly_periods (list): Anomaly periods.
        output_dir (str): Output directory.
        n_processes (int, optional): Number of processes (default: CPU cores, capped by file count).
    """
    print("=== Start extracting anomaly trace data (multiprocessing) ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List trace files
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    print(f"Found {len(trace_files)} trace files")
    
    # Determine number of processes
    if n_processes is None:
        n_processes = min(cpu_count(), len(trace_files))  # do not exceed file count
    
    print(f"Using {n_processes} processes")
    
    # Prepare argument list
    args_list = [(trace_file, trace_dir, anomaly_periods, output_dir) for trace_file in trace_files]
    
    # Run multiprocessing
    start_time = datetime.now()
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_trace_file, args_list)
    
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
        anomaly_periods (list): Anomaly periods (triples). Only the start timestamp is used for bucketing.
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
    for period_idx, (start_time, end_time, _) in enumerate(tqdm(anomaly_periods, desc="Processing anomaly windows", unit="period")):
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
                    # Filter rows within this segment.
                    mask = (df['start_time_ts'] >= segment_start) & (df['start_time_ts'] < segment_end)
                    segment_data = df[mask]
                    
                    # If this segment has data, append to results.
                    if len(segment_data) > 0:
                        durations = segment_data['duration'].tolist()
                        
                        # Use full instance name as service key (e.g., dbservice1, dbservice2).
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
    # Initialize all timestamps as not selected (False).
    is_in_target = pd.Series(False, index=times.index)
    
    # Mark timestamps in any target window as selected (True).
    for start_time, end_time, _ in target_periods:
        # Find timestamps within this window.
        in_current_period = (times >= start_time) & (times <= end_time)
        
        # Mark as selected.
        is_in_target.loc[in_current_period] = True
    
    return is_in_target

def main():
    """
    Entry point.
    """
    # Define file paths (relative to project root).
    label_file = os.path.join(_project_root, "data", "processed_data", "gaia", "label_gaia.csv")
    trace_dir = os.path.join(_project_root, "data", "raw_data", "gaia", "trace")
    output_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "trace")
    
    # 1) Load anomaly periods
    anomaly_periods = load_anomaly_periods(label_file)

    # Extract anomaly trace data (multiprocessing)
    extract_anomaly_trace_data_multiprocess(trace_dir, anomaly_periods, output_dir)
    
    """
    # 8) Extract per-service duration time series (optional)
    # json_output_file = os.path.join(output_dir, "service_durations_timeseries.json")
    # extract_service_durations_by_timesegments(output_dir, anomaly_periods, json_output_file)  # note: output may have overlapping timestamps
    """


if __name__ == "__main__":
    main()
