import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
import pickle

# Module-level private paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)

from utils.template_utils import get_log_template_count

# 1) Global configuration
SERVICES = [
    'compose-post-service', 'home-timeline-service', 'media-service', 
    'nginx-web-server', 'post-storage-service', 'social-graph-service', 
    'text-service', 'unique-id-service', 'url-shorten-service', 
    'user-mention-service', 'user-service', 'user-timeline-service'
]

# Metric column names (verified from CSV)
METRIC_COLUMNS = [
    'cpu_usage_system', 'cpu_usage_total', 'cpu_usage_user', 
    'memory_usage', 'memory_working_set', 'rx_bytes', 'tx_bytes'
]

NUM_METRICS = len(METRIC_COLUMNS)  # 7 (excluding timestamp)
NUM_INSTANCES = len(SERVICES)

# ⚠️ Important: must match window_size in process_sn_label.py
NUM_TIME_STEPS = 10  # seconds
STEP_DURATION = 1  # seconds
# Log template count (dynamically inferred)
NUM_LOG_TEMPLATES = get_log_template_count('sn')

# Global caches
METRIC_DATA_CACHE = {}
LOG_DATA_CACHE = {}
TRACE_DATA_CACHE = {}

# Normalization stats
NORMALIZATION_STATS = {
    'metric': None,
    'log': None,
    'trace': None
}

def preload_all_data():
    """
    Preload all modalities into memory.

    Note: SN timestamps are already in seconds (int/float). Unlike Gaia, do NOT divide by 10**6
    and do NOT multiply by 1000.
    """
    print("=" * 50)
    print("Preloading all SN data into memory...")
    print("=" * 50)
    
    # 1. Metric
    print("\n[1/3] Loading metric data...")
    metric_dir = os.path.join(_project_root, 'data', 'processed_data', 'sn', 'metric')
    for instance_name in tqdm(SERVICES, desc="Metric"):
        fpath = os.path.join(metric_dir, f"{instance_name}_metric.csv")
        if os.path.exists(fpath):
            try:
                # Read all columns; strip potential whitespace in column names.
                df = pd.read_csv(fpath, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                METRIC_DATA_CACHE[instance_name] = df
            except Exception as e:
                print(f"❌ Error reading {instance_name} metric: {e}")
    
    # 2. Log
    print("\n[2/3] Loading log data...")
    log_dir = os.path.join(_project_root, 'data', 'processed_data', 'sn', 'log')
    for instance_name in tqdm(SERVICES, desc="Log"):
        fpath = os.path.join(log_dir, f"{instance_name}_log.csv")
        if os.path.exists(fpath):
            try:
                # Read all columns; strip whitespace in column names.
                df = pd.read_csv(fpath, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                
                if 'timestamp' in df.columns and 'template_id' in df.columns:
                    LOG_DATA_CACHE[instance_name] = df[['timestamp', 'template_id']]
                else:
                    print(f"⚠️  Warning: {instance_name} log file missing columns. Found: {df.columns.tolist()}")
            except Exception as e:
                print(f"❌ Error reading {instance_name} log: {e}")
            
    # 3. Trace
    print("\n[3/3] Loading trace data...")
    trace_dir = os.path.join(_project_root, 'data', 'processed_data', 'sn', 'trace')
    for instance_name in tqdm(SERVICES, desc="Trace"):
        fpath = os.path.join(trace_dir, f"{instance_name}_trace.csv")
        if os.path.exists(fpath):
            try:
                # Read all columns; strip whitespace in column names.
                df = pd.read_csv(fpath, skipinitialspace=True)
                df.columns = df.columns.str.strip()
                
                required = ['start_time_ts', 'duration', 'status_code']
                if all(col in df.columns for col in required):
                    TRACE_DATA_CACHE[instance_name] = df[required]
                else:
                    print(f"⚠️  Warning: {instance_name} trace file missing columns. Found: {df.columns.tolist()}")
            except Exception as e:
                print(f"❌ Error reading {instance_name} trace: {e}")
            
    print(f"\nPreload complete: Metric({len(METRIC_DATA_CACHE)}), Log({len(LOG_DATA_CACHE)}), Trace({len(TRACE_DATA_CACHE)})")

def compute_normalization_stats(label_df):
    """
    Compute normalization statistics from training samples.
    """
    train_df = label_df[label_df['data_type'] == 'train']
    print(f"\nComputing statistics from {len(train_df)} training samples...")
    
    all_metrics = [[] for _ in range(NUM_METRICS)]
    all_logs = [[] for _ in range(NUM_LOG_TEMPLATES)]
    all_traces = [[] for _ in range(NUM_INSTANCES)]  # Trace duration per service
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Collecting training data"):
        # Timestamp handling (seconds) - explicitly use UTC.
        st_time = pd.to_datetime(row['st_time'], utc=True).timestamp()
        # 10-second window
        ed_time = st_time + (NUM_TIME_STEPS * STEP_DURATION)
        
        # Get raw data (normalize=False)
        metric, _ = _process_metric_for_sample(st_time, ed_time, normalize=False)
        log, _ = _process_log_for_sample(st_time, ed_time, normalize=False)
        trace, _ = _process_trace_for_sample(st_time, ed_time, normalize=False)
        
        # Metric: collect non-NaN values
        for i in range(NUM_METRICS):
            vals = metric[:, :, i].flatten()
            all_metrics[i].extend(vals[~np.isnan(vals)])
            
        # Log: collect non-zero values
        for i in range(NUM_LOG_TEMPLATES):
            vals = log[:, i].flatten()
            all_logs[i].extend(vals[vals != 0])
            
        # Trace duration: collect non-NaN and non-zero values
        for i in range(NUM_INSTANCES):
            vals = trace[i, :, 0]  # Ch0: Duration
            all_traces[i].extend(vals[~np.isnan(vals) & (vals != 0)])
            
    # Compute mean/std
    metric_stats = {'mean': np.zeros(NUM_METRICS), 'std': np.ones(NUM_METRICS)}
    for i in range(NUM_METRICS):
        if all_metrics[i]:
            metric_stats['mean'][i] = np.mean(all_metrics[i])
            metric_stats['std'][i] = np.std(all_metrics[i]) or 1.0
            
    log_stats = {'mean': np.zeros(NUM_LOG_TEMPLATES), 'std': np.ones(NUM_LOG_TEMPLATES)}
    for i in range(NUM_LOG_TEMPLATES):
        if all_logs[i]:
            log_stats['mean'][i] = np.mean(all_logs[i])
            log_stats['std'][i] = np.std(all_logs[i]) or 1.0
            
    trace_stats = []
    for i in range(NUM_INSTANCES):
        mean, std = 0.0, 1.0
        if all_traces[i]:
            mean = np.mean(all_traces[i])
            std = np.std(all_traces[i]) or 1.0
        trace_stats.append({'mean': mean, 'std': std})
        
    print("Statistics computed.")
    return {'metric': metric_stats, 'log': log_stats, 'trace': trace_stats}

def _process_metric_for_sample(st_time, ed_time, normalize=True):
    """
    Process metric features for a single sample (using preloaded caches).
    
    Args:
        st_time: Fault start timestamp (seconds).
        ed_time: Fault end timestamp (seconds).
        normalize: Whether to normalize (default: True).
    
    Returns:
        tuple: (metric_data, availability)
            - metric_data: numpy array, shape [12, 10, 7]
            - availability: bool - whether the metric modality is available
    """
    metric_data = np.full((NUM_INSTANCES, NUM_TIME_STEPS, NUM_METRICS), np.nan)
    
    for instance_idx, instance_name in enumerate(SERVICES):
        if instance_name not in METRIC_DATA_CACHE:
            continue
            
        df = METRIC_DATA_CACHE[instance_name]
        # Time filter [st, ed)
        mask = (df['timestamp'] >= st_time) & (df['timestamp'] < ed_time)
        sample = df[mask].sort_values('timestamp')
        
        if not sample.empty:
            # Align to 1-second steps.
            # sample['timestamp'] should be integer seconds.
            # Compute relative indices.
            rel_idx = (sample['timestamp'] - st_time).astype(int)
            valid_mask = (rel_idx >= 0) & (rel_idx < NUM_TIME_STEPS)
            
            valid_sample = sample[valid_mask]
            valid_idx = rel_idx[valid_mask]
            
            # Fill data
            metric_data[instance_idx, valid_idx, :] = valid_sample[METRIC_COLUMNS].values
            
    # Linear interpolation / forward fill could be applied along the time axis.
    # For simplicity, when normalize=True we fill NaN with the mean.
    
    availability = not np.all(np.isnan(metric_data))
    
    if normalize and NORMALIZATION_STATS['metric']:
        stats = NORMALIZATION_STATS['metric']
        for metric_idx in range(NUM_METRICS):
            nan_mask = np.isnan(metric_data[:, :, metric_idx])
            if nan_mask.any():
                metric_data[:, :, metric_idx][nan_mask] = stats['mean'][metric_idx]
            metric_data[:, :, metric_idx] = (metric_data[:, :, metric_idx] - stats['mean'][metric_idx]) / stats['std'][metric_idx]
            
    return metric_data, availability

def _process_log_for_sample(st_time, ed_time, normalize=True):
    """
    Process log features for a single sample (using preloaded caches).
    
    Args:
        st_time: Fault start timestamp (seconds).
        ed_time: Fault end timestamp (seconds).
        normalize: Whether to normalize (default: True).
    
    Returns:
        tuple: (log_data, availability)
            - log_data: numpy array, shape [12, 13]
            - availability: bool - whether the log modality is available
    """
    log_data = np.zeros((NUM_INSTANCES, NUM_LOG_TEMPLATES))
    
    for instance_idx, instance_name in enumerate(SERVICES):
        if instance_name not in LOG_DATA_CACHE:
            continue
            
        df = LOG_DATA_CACHE[instance_name]
        # SN log timestamps are float seconds
        mask = (df['timestamp'] >= st_time) & (df['timestamp'] < ed_time)
        sample_data = df[mask]
        
        if not sample_data.empty:
            template_counts = sample_data['template_id'].value_counts()
            for template_id, count in template_counts.items():
                # template_id starts at 1; array index starts at 0
                if 1 <= template_id <= NUM_LOG_TEMPLATES:
                    log_data[instance_idx, template_id - 1] = count
                    
    availability = not np.all(log_data == 0)
    
    if normalize and NORMALIZATION_STATS['log']:
        stats = NORMALIZATION_STATS['log']
        log_data = (log_data - stats['mean']) / stats['std']
        
    return log_data, availability

def _process_trace_for_sample(st_time, ed_time, normalize=True):
    """
    Process trace features for a single sample (using preloaded caches).
    
    Two-channel feature extraction:
    - Channel 0: Duration (latency)
    - Channel 1: Error Rate (based on status_code >= 400)
    
    Args:
        st_time: Fault start timestamp (seconds).
        ed_time: Fault end timestamp (seconds).
        normalize: Whether to normalize (default: True).
    
    Returns:
        tuple: (trace_data, availability)
            - trace_data: numpy array, shape [12, 10, 2]
            - availability: bool - whether the trace modality is available
    """
    trace_data = np.full((NUM_INSTANCES, NUM_TIME_STEPS, 2), np.nan)
    
    for instance_idx, instance_name in enumerate(SERVICES):
        if instance_name not in TRACE_DATA_CACHE:
            continue
            
        df = TRACE_DATA_CACHE[instance_name]
        # Time filter
        mask = (df['start_time_ts'] >= st_time) & (df['start_time_ts'] < ed_time)
        sample_data = df[mask]
        
        if not sample_data.empty:
            timestamps = sample_data['start_time_ts'].values
            durations = sample_data['duration'].values
            status_codes = sample_data['status_code'].values
            
            # Compute time-step indices (0..NUM_TIME_STEPS-1)
            segment_indices = ((timestamps - st_time) // STEP_DURATION).astype(int)
            
            # Aggregate per step
            for seg_idx in range(NUM_TIME_STEPS):
                seg_mask = segment_indices == seg_idx
                if seg_mask.any():
                    # Duration Mean
                    trace_data[instance_idx, seg_idx, 0] = np.mean(durations[seg_mask])
                    # Error Rate
                    seg_status = status_codes[seg_mask]
                    error_count = np.sum(seg_status >= 400)
                    error_rate = error_count / len(seg_status)
                    trace_data[instance_idx, seg_idx, 1] = error_rate
                    
    availability = not np.all(np.isnan(trace_data))
    
    if normalize and NORMALIZATION_STATS['trace']:
        # Process all services
        for instance_idx in range(NUM_INSTANCES):
            instance_name = SERVICES[instance_idx]
            
            if instance_name in TRACE_DATA_CACHE:
                # Services with trace data: normalize normally.
                stats = NORMALIZATION_STATS['trace'][instance_idx]  # use service index
                
                # Duration: Fill Mean -> Normalize
                nan_mask_0 = np.isnan(trace_data[instance_idx, :, 0])
                if nan_mask_0.any():
                    trace_data[instance_idx, :, 0][nan_mask_0] = stats['mean']
                trace_data[instance_idx, :, 0] = (trace_data[instance_idx, :, 0] - stats['mean']) / stats['std']
                
                # ErrorRate: Fill 0 -> No Normalize
                nan_mask_1 = np.isnan(trace_data[instance_idx, :, 1])
                if nan_mask_1.any():
                    trace_data[instance_idx, :, 1][nan_mask_1] = 0.0
            else:
                # Services without trace data: fill defaults (no call activity).
                trace_data[instance_idx, :, 0] = 0.0  # Duration = 0 (no calls)
                trace_data[instance_idx, :, 1] = 0.0  # Error rate = 0 (no errors)
                
    return trace_data, availability

def _process_single_sample(row) -> Dict[str, Any]:
    """
    Process a single fault sample.
    """
    sample_id = row['index']
    fault_service = row['instance']
    fault_type = row['anomaly_type']
    st_time = pd.to_datetime(row['st_time'], utc=True).timestamp()
    ed_time = st_time + (NUM_TIME_STEPS * STEP_DURATION)
    data_type = row['data_type']

    processed_sample = {
        'sample_id': sample_id,
        'fault_service': fault_service,
        'fault_type': fault_type,
        'st_time': st_time,
        'ed_time': ed_time,
        'data_type': data_type,
    }

    # Process each modality (returns features and modality-level availability flags).
    metric_data, metric_available = _process_metric_for_sample(st_time, ed_time)
    log_data, log_available = _process_log_for_sample(st_time, ed_time)
    trace_data, trace_available = _process_trace_for_sample(st_time, ed_time)
    
    processed_sample['metric_data'] = metric_data
    processed_sample['log_data'] = log_data
    processed_sample['trace_data'] = trace_data
    
    # Add modality-level availability flags.
    processed_sample['metric_available'] = metric_available  # bool
    processed_sample['log_available'] = log_available        # bool
    processed_sample['trace_available'] = trace_available    # bool
    
    return processed_sample


def process_all_sample(label_df) -> Dict[int, Dict[str, Any]]:
    """
    Process all fault samples.
    """
    processed_data = {}
    
    print(f"\nProcessing {len(label_df)} fault samples...")
    
    for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Processing samples"):
        sample_id = row['index']
        try:
            processed_sample = _process_single_sample(row)
            processed_data[sample_id] = processed_sample
        except Exception as e:
            print(f"\n❌ Failed to process sample {sample_id}: {e}")
            continue
    
    print(f"\n✅ Done! Successfully processed {len(processed_data)}/{len(label_df)} samples")
    return processed_data

if __name__ == "__main__":
    label_path = os.path.join(_project_root, "data", "processed_data", "sn", "label_sn.csv")
    label_df = pd.read_csv(label_path)
    
    # 1. Preload
    preload_all_data()
    
    # 2. Stats
    stats_file = os.path.join(_project_root, "data", "processed_data", "sn", "sn_norm_stats.pkl")
    if os.path.exists(stats_file):
        print(f"Loading stats from {stats_file}")
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
    else:
        stats = compute_normalization_stats(label_df)
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
            
    NORMALIZATION_STATS['metric'] = stats['metric']
    NORMALIZATION_STATS['log'] = stats['log']
    NORMALIZATION_STATS['trace'] = stats['trace']
    
    # 3. Process
    dataset = process_all_sample(label_df)
    
    # 4. Save
    out_path = os.path.join(_project_root, "data", "processed_data", "sn", "dataset.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"\nDataset saved to {out_path}")
    print(f"Total samples: {len(dataset)}")
