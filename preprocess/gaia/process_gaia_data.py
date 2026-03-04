import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any

# Add project root to path for importing local modules.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)

from utils.template_utils import get_log_template_count

# Global: fixed instance order (used by all modalities).
SERVICES = ['dbservice1', 'dbservice2', 'logservice1', 'logservice2', 
            'mobservice1', 'mobservice2', 'redisservice1', 'redisservice2', 
            'webservice1', 'webservice2']

# Global caches: preloaded data.
METRIC_DATA_CACHE = {}
LOG_DATA_CACHE = {}
TRACE_DATA_CACHE = {}

# Global normalization stats (computed from training samples).
NORMALIZATION_STATS = {
    'metric': None,  # {'mean': [12], 'std': [12]}
    'log': None,     # {'mean': [48], 'std': [48]}
    'trace': None    # [{'mean': float, 'std': float}] * 10 (for duration only)
}


def preload_all_data():
    """
    Preload all modalities into memory.
    """
    print("=" * 50)
    print("Preloading all data into memory...")
    print("=" * 50)
    
    # 1) Metric
    print("\n[1/3] Loading metric data...")
    metric_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    for instance_name in tqdm(SERVICES, desc="Metric"):
        metric_file = os.path.join(metric_data_dir, f"{instance_name}_metric.csv")
        if os.path.exists(metric_file):
            df = pd.read_csv(metric_file)
            METRIC_DATA_CACHE[instance_name] = df
            print(f"  ✓ {instance_name}: {len(df)} rows")
    
    # 2) Log
    print("\n[2/3] Loading log data...")
    log_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'log')
    for instance_name in tqdm(SERVICES, desc="Log"):
        log_file = os.path.join(log_data_dir, f"{instance_name}_log.csv")
        if os.path.exists(log_file):
            # Only load required columns to save memory.
            df = pd.read_csv(log_file, usecols=['timestamp_ts', 'template_id'])
            LOG_DATA_CACHE[instance_name] = df
            print(f"  ✓ {instance_name}: {len(df)} rows")
    
    # 3) Trace
    print("\n[3/3] Loading trace data (with status_code)...")
    trace_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'trace')
    for instance_name in tqdm(SERVICES, desc="Trace"):
        trace_file = os.path.join(trace_data_dir, f"{instance_name}_trace.csv")
        if os.path.exists(trace_file):
            # Load duration and status_code.
            df = pd.read_csv(trace_file, usecols=['start_time_ts', 'duration', 'status_code'])
            TRACE_DATA_CACHE[instance_name] = df
            print(f"  ✓ {instance_name}: {len(df)} rows")
    
    print("\n" + "=" * 50)
    print("Preloading finished.")
    print(f"  Metric instances: {len(METRIC_DATA_CACHE)}")
    print(f"  Log instances: {len(LOG_DATA_CACHE)}")
    print(f"  Trace instances: {len(TRACE_DATA_CACHE)}")
    print("=" * 50 + "\n")


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
    
    # Convert start time to timestamp (ms); each anomaly window lasts 600 seconds.
    anomaly_periods = []
    for _, row in label_df.iterrows():
        # Convert start time string to timestamp (ms).
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000  # start + 600 seconds
        data_type = row.get('data_type', 'unknown')  # default: unknown
        anomaly_periods.append((st_time, ed_time, data_type))
    
    print(f"Loaded {len(anomaly_periods)} anomaly periods")
    return anomaly_periods


def compute_normalization_stats(label_df):
    """
    Compute normalization statistics from training samples.
    
    Rules:
        - Metric: exclude NaN; keep 0 (0 is a real value).
        - Log: exclude 0 (0 means "not occurred").
        - Trace: exclude NaN and 0 (both represent missing); only compute stats for Duration.
    """
    train_df = label_df[label_df['data_type'] == 'train']
    print(f"\nComputing normalization stats from {len(train_df)} training samples...")
    
    # Collect values per dimension.
    all_metrics = [[] for _ in range(12)]  # 12 metric dimensions
    all_logs = [[] for _ in range(48)]     # 48 log templates
    all_traces = [[] for _ in range(10)]   # 10 instances (duration only)
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Collecting training data"):
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000
        
        # Collect raw data (no normalization).
        metric, _ = _process_metric_for_sample(st_time, ed_time, normalize=False)
        log, _ = _process_log_for_sample(st_time, ed_time, normalize=False)
        trace, _ = _process_trace_for_sample(st_time, ed_time, normalize=False)
        
        # Metric: keep non-NaN values (including 0).
        for i in range(12):
            vals = metric[:, :, i].flatten()
            valid_vals = vals[~np.isnan(vals)]  # exclude NaN, keep 0
            all_metrics[i].extend(valid_vals)
        
        # Log: keep non-zero values.
        for i in range(48):
            vals = log[:, i].flatten()
            non_zero_vals = vals[vals != 0]  # exclude 0
            all_logs[i].extend(non_zero_vals)
        
        # Trace: keep non-NaN and non-zero values (Channel 0: Duration only).
        for i in range(10):
            vals = trace[i, :, 0]  # Channel 0: Duration
            valid_vals = vals[~np.isnan(vals) & (vals != 0)]  # exclude NaN and 0
            all_traces[i].extend(valid_vals)
    
    # Compute mean/std per dimension.
    print("\nComputing statistics:")
    
    # Metric stats
    metric_means = np.zeros(12)
    metric_stds = np.zeros(12)
    for i in range(12):
        if len(all_metrics[i]) > 0:
            metric_means[i] = np.mean(all_metrics[i])
            metric_stds[i] = np.std(all_metrics[i])
            if metric_stds[i] == 0:
                metric_stds[i] = 1.0
            print(f"  Metric[{i}]: mean={metric_means[i]:.4f}, std={metric_stds[i]:.4f}, samples={len(all_metrics[i])}")
        else:
            metric_means[i] = 0.0
            metric_stds[i] = 1.0
            print(f"  Metric[{i}]: no valid data")
    
    # Log stats
    log_means = np.zeros(48)
    log_stds = np.zeros(48)
    for i in range(48):
        if len(all_logs[i]) > 0:
            log_means[i] = np.mean(all_logs[i])
            log_stds[i] = np.std(all_logs[i])
            if log_stds[i] == 0:
                log_stds[i] = 1.0
        else:
            log_means[i] = 0.0
            log_stds[i] = 1.0
    print(f"  Log: {np.sum([len(all_logs[i]) > 0 for i in range(48)])}/48 templates have data")
    
    # Trace stats (Duration only)
    trace_stats = []
    for i in range(10):
        if len(all_traces[i]) > 0:
            mean, std = np.mean(all_traces[i]), np.std(all_traces[i])
            trace_stats.append({'mean': mean, 'std': std if std > 0 else 1.0})
            print(f"  Trace[{SERVICES[i]}]: mean={mean:.4f}, std={std:.4f}, samples={len(all_traces[i])}")
        else:
            trace_stats.append({'mean': 0.0, 'std': 1.0})
            print(f"  Trace[{SERVICES[i]}]: no valid data")
    
    metric_stats = {'mean': metric_means, 'std': metric_stds}
    log_stats = {'mean': log_means, 'std': log_stds}
    
    print("\n✅ Statistics computed.")
    return {'metric': metric_stats, 'log': log_stats, 'trace': trace_stats}


def _process_metric_for_sample(st_time, ed_time, normalize=True):
    """
    Process metric features for a single sample (using preloaded caches).
    
    Args:
        st_time: Start timestamp (ms).
        ed_time: End timestamp (ms).
        normalize: Whether to normalize (default: True).
    
    Returns:
        tuple: (metric_data, availability)
            - metric_data: numpy array, shape [10, 20, 12]
            - availability: bool - whether the metric modality is available
    """    
    # Use global instance order.
    num_instances = len(SERVICES)
    
    # Initialize [num_instances, 20 time_steps, 12 metrics] with NaN to mark missing values.
    metric_data = np.full((num_instances, 20, 12), np.nan)
    metric_names = None
    
    # Iterate each instance in the fixed order.
    for instance_idx, instance_name in enumerate(SERVICES):
        # Read from cache.
        if instance_name not in METRIC_DATA_CACHE:
            continue
        
        try:
            df = METRIC_DATA_CACHE[instance_name]
            mask = (df['timestamp'] >= st_time) & (df['timestamp'] <= ed_time)
            sample_data = df[mask].sort_values('timestamp')
            
            if metric_names is None:
                metric_names = [col for col in sample_data.columns if col != 'timestamp']
            
            # Assign all metric columns at once.
            num_time_steps = min(len(sample_data), 20)
            if num_time_steps > 0:
                metric_data[instance_idx, :num_time_steps, :] = sample_data[metric_names].values[:num_time_steps]
        
        except Exception:
            continue
    
    # Modality availability: if all values are NaN, the whole modality is unavailable.
    availability = not np.all(np.isnan(metric_data))
    
    # Fill missing values and normalize.
    if normalize and NORMALIZATION_STATS['metric'] is not None:
        stats = NORMALIZATION_STATS['metric']
        
        # Fill NaN with mean.
        for i in range(12):  # 12 metrics
            nan_mask = np.isnan(metric_data[:, :, i])
            if nan_mask.any():
                metric_data[:, :, i][nan_mask] = stats['mean'][i]
        
        # Normalize.
        metric_data = (metric_data - stats['mean']) / stats['std']
    else:
        # For stats-collection stage (normalize=False), keep NaN as-is.
        pass
    
    return metric_data, availability

def _process_log_for_sample(st_time, ed_time, normalize=True):
    """
    Process log features for a single sample (using preloaded caches).
    
    Args:
        st_time: Start timestamp (ms).
        ed_time: End timestamp (ms).
        normalize: Whether to normalize (default: True).
    
    Returns:
        tuple: (log_data, availability)
            - log_data: numpy array, shape [10, num_templates]
            - availability: bool - whether the log modality is available
    """
    # Use global instance order.
    num_instances = len(SERVICES)
    num_templates = get_log_template_count('gaia')  # dynamically infer number of templates
    
    # Initialize [num_instances, num_templates].
    log_data = np.zeros((num_instances, num_templates))
    
    # Iterate each instance in the fixed order.
    for instance_idx, instance_name in enumerate(SERVICES):
        # Read from cache.
        if instance_name not in LOG_DATA_CACHE:
            continue
        
        try:
            df = LOG_DATA_CACHE[instance_name]
            
            # Filter by time window.
            mask = (df['timestamp_ts'] >= st_time) & (df['timestamp_ts'] <= ed_time)
            sample_data = df[mask]
            
            if len(sample_data) > 0:
                # Count occurrences per template_id.
                template_counts = sample_data['template_id'].value_counts()
                
                # Fill counts (template_id starts from 1; array index starts from 0).
                for template_id, count in template_counts.items():
                    if 1 <= template_id <= num_templates:
                        log_data[instance_idx, template_id - 1] = count
        
        except Exception:
            continue
    
    # Modality availability: if all values are 0, the whole modality is unavailable.
    availability = not np.all(log_data == 0)
    
    # Normalize (do not fill; keep 0s as-is).
    if normalize and NORMALIZATION_STATS['log'] is not None:
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
        st_time: Start timestamp (ms).
        ed_time: End timestamp (ms).
        normalize: Whether to normalize (default: True).
    
    Returns:
        tuple: (trace_data, availability)
            - trace_data: numpy array, shape [10, 20, 2]
            - availability: bool - whether the trace modality is available
    """
    # Use global instance order.
    num_instances = len(SERVICES)
    num_time_segments = 20  # 20 segments
    segment_duration = 30 * 1000  # 30 seconds per segment (ms)
    num_channels = 2  # Duration + ErrorRate
    
    # Initialize [num_instances, num_time_segments, 2] with NaN.
    trace_data = np.full((num_instances, num_time_segments, num_channels), np.nan)
    
    # Iterate each instance in the fixed order.
    for instance_idx, instance_name in enumerate(SERVICES):
        # Read from cache.
        if instance_name not in TRACE_DATA_CACHE:
            continue
        
        try:
            df = TRACE_DATA_CACHE[instance_name]
            
            # Filter by time window.
            mask = (df['start_time_ts'] >= st_time) & (df['start_time_ts'] <= ed_time)
            sample_data = df[mask]
            
            if len(sample_data) > 0:
                # Vectorized computation: compute segment indices for all traces.
                timestamps = sample_data['start_time_ts'].values
                durations = sample_data['duration'].values
                status_codes = sample_data['status_code'].values
                
                # Compute time offsets and segment indices.
                time_offsets = timestamps - st_time
                segment_indices = (time_offsets // segment_duration).astype(int)
                
                # Keep valid segment indices.
                valid_mask = (segment_indices >= 0) & (segment_indices < num_time_segments)
                valid_segments = segment_indices[valid_mask]
                valid_durations = durations[valid_mask]
                valid_status = status_codes[valid_mask]
                
                # Aggregate by segment.
                for seg_idx in range(num_time_segments):
                    seg_mask = valid_segments == seg_idx
                    if seg_mask.any():
                        # 1) Mean duration
                        mean_duration = valid_durations[seg_mask].mean()
                        trace_data[instance_idx, seg_idx, 0] = mean_duration
                        
                        # 2) Error rate (status_code >= 400)
                        seg_status = valid_status[seg_mask]
                        error_count = np.sum(seg_status >= 400)
                        error_rate = error_count / len(seg_status)
                        trace_data[instance_idx, seg_idx, 1] = error_rate
        
        except Exception:
            continue
    
    # Modality availability: if all values are NaN, the whole modality is unavailable.
    availability = not np.all(np.isnan(trace_data))
    
    # Fill missing values and normalize.
    if normalize and NORMALIZATION_STATS['trace'] is not None:
        for i in range(num_instances):
            stats = NORMALIZATION_STATS['trace'][i]
            
            # Channel 0 (Duration): fill NaN with mean, then normalize.
            nan_mask_0 = np.isnan(trace_data[i, :, 0])
            if nan_mask_0.any():
                trace_data[i, :, 0][nan_mask_0] = stats['mean']
            trace_data[i, :, 0] = (trace_data[i, :, 0] - stats['mean']) / stats['std']
            
            # Channel 1 (Error Rate): fill NaN with 0 (no requests => no errors), keep as-is (0~1).
            nan_mask_1 = np.isnan(trace_data[i, :, 1])
            if nan_mask_1.any():
                trace_data[i, :, 1][nan_mask_1] = 0.0
            # Error Rate does not require z-score normalization.
            
    else:
        # For stats-collection stage (normalize=False), keep NaN as-is.
        pass
    
    return trace_data, availability


def _process_single_sample(row) -> Dict[str, Any]:
    """
    Process a single fault case.
    """
    sample_id = row['index']
    fault_service = row['instance']
    fault_type = row['anomaly_type']
    st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
    ed_time = st_time + 600 * 1000  # start + 600 seconds
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
    Process all fault cases.
    """
    processed_data = {}
    
    print(f"\nProcessing {len(label_df)} fault cases...")
    
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
    label_file = os.path.join(_project_root, "data", "processed_data", "gaia", "label_gaia.csv")
    label_df = pd.read_csv(label_file)
    
    # 1) Preload all data into memory
    preload_all_data()
    
    # 2) Compute or load normalization stats
    stats_file = os.path.join(_project_root, "data", "processed_data", "gaia", "norm_stats.pkl")
    
    if os.path.exists(stats_file):
        print(f"\n📂 Loading normalization stats: {stats_file}")
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
    else:
        print("\n🔄 Computing normalization stats...")
        stats = compute_normalization_stats(label_df)
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        print(f"✅ Stats saved: {stats_file}")
    
    # Set global stats
    NORMALIZATION_STATS['metric'] = stats['metric']
    NORMALIZATION_STATS['log'] = stats['log']
    NORMALIZATION_STATS['trace'] = stats['trace']
    
    # 3) Process all samples
    processed_data = process_all_sample(label_df)
    
    # 4) Save processed dataset
    output_file = os.path.join(_project_root, "data", "processed_data", "gaia", "dataset.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\n💾 Dataset saved: {output_file}")
    print(f"   - Num samples: {len(processed_data)}")
    print("\nProcessing summary:")
    print("   Metric: stats exclude NaN; NaN filled with mean")
    print("   Log: stats exclude 0; no filling (0 means 'not occurred')")
    print("   Trace: two-channel (Duration, ErrorRate)")
    print("     - Ch0 (Duration): normalized; NaN filled with mean")
    print("     - Ch1 (ErrorRate): not normalized; NaN filled with 0")
    print("\nAvailability flags: each sample includes modality-level flags")
    print("   - metric_available: bool (whether metric modality is available)")
    print("   - log_available: bool (whether log modality is available)")
    print("   - trace_available: bool (whether trace modality is available)")
