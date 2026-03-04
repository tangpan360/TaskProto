import os
import subprocess
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from datetime import datetime


# Get project root directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))

def copy_valid_metric_files() -> None:
    """
    Copy selected raw metric files into processed_data.

    This function scans the raw metric folder (`data/raw_data/gaia/metric`) and copies files
    that contain both a target service name and a set of key Docker metric names into
    `data/processed_data/gaia/metric`.

    Returns:
        None
    """
    # print(os.path.dirname(os.path.abspath(__file__)))
    
    metric_dir = os.path.join(_project_root, 'data', 'raw_data', 'gaia', 'metric')
    # print(metric_dir)

    file_names = os.listdir(metric_dir)

    services = ['dbservice', 'mobservice', 'logservice', 'webservice', 'redisservice']
    
    docker_metrics = [
        "docker_cpu_total_norm_pct",    # total CPU usage (normalized)
        # "docker_cpu_user_pct",          # user-mode CPU usage
        # "docker_cpu_kernel_pct",        # kernel-mode CPU usage
        "docker_memory_usage_pct",      # memory usage percentage
        # "docker_memory_usage_total",    # total memory usage
        # "docker_memory_limit",          # memory limit
        "docker_memory_fail_count",     # memory allocation failures
        "docker_diskio_read_bytes",     # disk read bytes
        "docker_diskio_write_bytes",    # disk write bytes
        # "docker_diskio_read_ops",       # disk read ops
        # "docker_diskio_write_ops",      # disk write ops
        "docker_diskio_read_service_time",  # disk read service time
        "docker_diskio_write_service_time", # disk write service time
        "docker_network_in_bytes",      # inbound bytes
        "docker_network_out_bytes",     # outbound bytes
        # "docker_network_in_packets",    # inbound packets
        # "docker_network_out_packets",   # outbound packets
        "docker_network_in_errors",     # inbound errors
        "docker_network_out_errors",    # outbound errors
        # "docker_network_in_dropped",    # inbound dropped packets
        "docker_network_out_dropped",   # outbound dropped packets
    ]
    
    # Keep files that include both a target service name and a key metric name.
    valid_files = []
    for fn in file_names:
        # Check whether the filename contains any service name.
        has_service = any(s in fn for s in services)
        # Check whether the filename contains any key metric name.
        has_metric = any(m in fn for m in docker_metrics)
        
        if has_service and has_metric:
            valid_files.append(fn)
    
    print(f"Selected {len(valid_files)} files with key metrics out of {len(file_names)} total files")
    
    # Copy selected files into processed_data.
    for file in valid_files:
        file_path = os.path.join(metric_dir, file)
        processed_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        target_path = os.path.join(processed_data_dir, file)
        subprocess.run(['cp', file_path, target_path])

def merge_date_range_files() -> None:
    """
    Merge date-range files for the same service/host/metric.
    
    Example:
        Merge 2021-07-01_2021-07-15 and 2021-07-15_2021-07-31 into 2021-07-01_2021-07-31.

    Returns:
        None
    """
    processed_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    
    if not os.path.exists(processed_data_dir):
        print("processed_data directory does not exist")
        return
    
    # Collect all 2021-07-01_2021-07-15 files.
    first_period_files = []
    for f in os.listdir(processed_data_dir):
        if "2021-07-01_2021-07-15" in f and f.endswith('.csv'):
            first_period_files.append(f)
    
    print(f"Found {len(first_period_files)} files in the first period")
    
    merged_count = 0
    failed_count = 0
    
    for first_file in tqdm(first_period_files, desc="Merging date-range files"):
        # Construct corresponding second-period filename.
        second_file = first_file.replace("2021-07-01_2021-07-15", "2021-07-15_2021-07-31")
        merged_file = first_file.replace("2021-07-01_2021-07-15", "2021-07-01_2021-07-31")
        
        first_path = os.path.join(processed_data_dir, first_file)
        second_path = os.path.join(processed_data_dir, second_file)
        merged_path = os.path.join(processed_data_dir, merged_file)
        
        # Check if the second file exists.
        if not os.path.exists(second_path):
            print(f"Warning: missing corresponding second-period file: {second_file}")
            failed_count += 1
            continue
        
        try:
            # Read both CSV files.
            df1 = pd.read_csv(first_path)
            df2 = pd.read_csv(second_path)
            
            # Merge
            merged_df = pd.concat([df1, df2], ignore_index=True)
            
            # Sort by timestamp.
            timestamp_col = 'timestamp'
            merged_df = merged_df.sort_values(timestamp_col).reset_index(drop=True)
            
            # Drop duplicate timestamps, keeping the first occurrence.
            merged_df = merged_df.drop_duplicates(subset=[timestamp_col], keep='first').reset_index(drop=True)
            
            # Save merged file.
            merged_df.to_csv(merged_path, index=False)
            
            # Remove original files.
            os.remove(first_path)
            os.remove(second_path)
            
            merged_count += 1
            
        except Exception as e:
            print(f"Error merging {first_file}: {e}")
            failed_count += 1
            continue
    
    print(f"Successfully merged {merged_count} file pairs")
    print(f"Failed merges: {failed_count} file pairs")
    print(f"Total CSV files after merging: {len([f for f in os.listdir(processed_data_dir) if f.endswith('.csv')])}")

def merge_metrics_by_service_instance() -> None:
    """
    Merge per-metric CSV files into a multi-column CSV per service instance.

    Example:
    - dbservice1_0.0.0.4_docker_cpu_kernel_pct_2021-07-01_2021-07-31.csv
    - dbservice1_0.0.0.4_docker_cpu_user_pct_2021-07-01_2021-07-31.csv
    Output:
    - dbservice1_metric.csv (with multiple metric columns)
    
    Returns:
        None
    """
    processed_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    
    # List CSV files.
    csv_files = os.listdir(processed_data_dir)
    csv_files = sorted(csv_files)
    print(f"Found {len(csv_files)} metric files to merge")
    
    # Group files by service instance.
    service_groups = {}
    for filename in csv_files:
        # Filename pattern: service_ip_metric_daterange.csv
        # Example: dbservice1_0.0.0.4_docker_cpu_kernel_pct_2021-07-01_2021-07-31.csv
        
        # Split by "_"
        splits = filename.replace('.csv', '').split('_')
        
        if len(splits) >= 6:  # At least: service_ip_docker_metric_2021-07-01_2021-07-31
            # Use the first part as service instance (e.g., dbservice1).
            service_instance = splits[0]
            
            # Metric name is from the 4th part up to (but excluding) the last 2 date parts.
            # Example: ['dbservice1','0.0.0.4','docker','cpu','kernel','pct','2021-07-01','2021-07-31']
            # Metric name: splits[3:len(splits)-2] -> ['cpu','kernel','pct']
            metric_name = '_'.join(splits[3:len(splits)-2])
            
            if service_instance not in service_groups:
                service_groups[service_instance] = {}
            
            service_groups[service_instance][metric_name] = {
                'filename': filename
            }
        else:
            print(f"⚠️  Failed to parse filename: {filename} (only {len(splits)} parts after split)")
    
    print(f"Identified {len(service_groups)} service instances")
    
    # Collect and sort all metric names to keep a consistent column order across instances.
    all_metrics = set()
    for metrics in service_groups.values():
        all_metrics.update(metrics.keys())
    sorted_metrics = sorted(all_metrics)
    print(f"Identified {len(sorted_metrics)} distinct metrics")
    print(f"Metric list (alphabetical): {sorted_metrics}")
    
    merged_count = 0
    
    for service_instance, metrics in service_groups.items():
        
        print(f"🔄 Merging {service_instance}: {len(metrics)} metrics")
        
        try:
            merged_df = None
            
            for metric_name, info in metrics.items():
                file_path = os.path.join(processed_data_dir, info['filename'])
                df = pd.read_csv(file_path)
                
                # Rename value column to the metric name.
                df = df.rename(columns={'value': metric_name})
                
                if merged_df is None:
                    merged_df = df
                else:
                    # Outer join by timestamp.
                    merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')
            
            # Sort by timestamp.
            merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
            
            # Keep a consistent column order: timestamp first, then metrics in alphabetical order.
            # Only keep metrics that exist for this instance.
            available_metrics = [m for m in sorted_metrics if m in merged_df.columns]
            column_order = ['timestamp'] + available_metrics
            merged_df = merged_df[column_order]
            
            # Output filename.
            merged_filename = f"{service_instance}_metric.csv"
            merged_path = os.path.join(processed_data_dir, merged_filename)
            
            # Save merged CSV.
            merged_df.to_csv(merged_path, index=False)
            
            # Remove original per-metric files.
            for metric_name, info in metrics.items():
                original_path = os.path.join(processed_data_dir, info['filename'])
                os.remove(original_path)
            
            print(f"✅ {service_instance}: merge complete -> {merged_filename}")
            print(f"   Metrics: {list(metrics.keys())}")
            print(f"   Rows: {len(merged_df)}")
            
            merged_count += 1
            
        except Exception as e:
            print(f"❌ Error merging {service_instance}: {e}")
            continue
    
    print("\nMerge finished:")
    print(f"  Successfully merged instances: {merged_count}")
    print(f"  Final CSV file count: {len([f for f in os.listdir(processed_data_dir) if f.endswith('.csv')])}")


def process_single_file_resample(file_path: str) -> tuple:
    """
    Resample a single file to 30-second intervals and overwrite the original file.
    
    Args:
        file_path: Absolute file path.
    
    Returns:
        tuple: (filename, success, original_rows, resampled_rows, error_message_or_None)
    """
    try:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path)
        
        # Check for timestamp column.
        if 'timestamp' not in df.columns:
            return (filename, False, 0, 0, "missing timestamp column")
        
        original_rows = len(df)
        
        # Data time range
        min_timestamp = df['timestamp'].min()
        max_timestamp = df['timestamp'].max()
        
        # Fixed start timestamp (ms) and interval.
        start_timestamp = 1625133601000  # 2021-07-01 00:00:01 GMT, fixed start time
        interval_ms = 30 * 1000  # 30 seconds in ms
        
        # Start from the fixed start timestamp.
        current_timestamp = start_timestamp
        
        # Accumulate resampled rows.
        resampled_data = []
        
        while current_timestamp <= max_timestamp:
            # Define end of the 30-second window.
            window_end = current_timestamp + interval_ms
            
            # Find rows within this 30-second window.
            window_data = df[(df['timestamp'] >= current_timestamp) & 
                           (df['timestamp'] < window_end)]
            
            if not window_data.empty:
                # If there are rows, take the first (earliest).
                first_record = window_data.iloc[0].copy()
                # Snap timestamp to the window start.
                first_record['timestamp'] = current_timestamp
                resampled_data.append(first_record)
            else:
                # If there are no rows, create an empty record, keeping the timestamp.
                empty_record = pd.Series(index=df.columns)
                empty_record['timestamp'] = current_timestamp
                # Set all metric columns to NA.
                metric_columns = [col for col in df.columns if col != 'timestamp']
                for col in metric_columns:
                    empty_record[col] = pd.NA
                resampled_data.append(empty_record)
            
            # Move to the next 30-second window.
            current_timestamp += interval_ms
        
        if resampled_data:
            # Build DataFrame
            resampled_df = pd.DataFrame(resampled_data)
            
            # Overwrite the original file.
            resampled_df.to_csv(file_path, index=False)
            
            resampled_rows = len(resampled_df)
            return (filename, True, original_rows, resampled_rows, None)
        else:
            return (filename, False, original_rows, 0, "no data found within the specified time range")
            
    except Exception as e:
        return (os.path.basename(file_path), False, 0, 0, str(e))


def resample_metrics_30s_interval(num_processes: int = None) -> None:
    """
    Resample all processed metric files to 30-second intervals using multiprocessing.

    Starting from the fixed timestamp 1625133601000 (ms), check every 30 seconds:
    - If multiple values exist in a window, keep the first one and place it at the window start timestamp.
    - If no values exist, create an empty row for that timestamp.
    
    Args:
        num_processes: Number of processes to use (default: CPU cores, capped by file count).
    
    Returns:
        None
    """
    processed_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    
    # List CSV files.
    csv_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]
    
    # Build full file paths.
    file_paths = [os.path.join(processed_data_dir, file) for file in csv_files]
    
    # Determine process count.
    if num_processes is None:
        num_processes = min(cpu_count(), len(file_paths))
    
    print(f"Resampling {len(file_paths)} files at 30-second intervals (overwriting originals) using {num_processes} processes...")
    print("Fixed start timestamp: 1625133601000 (ms)")
    
    # Run multiprocessing pool.
    with Pool(processes=num_processes) as pool:
        # tqdm progress bar
        results = list(tqdm(
            pool.imap(process_single_file_resample, file_paths),
            total=len(file_paths),
            desc="Resampling files"
        ))
    
    # Summarize results
    successful_count = 0
    failed_count = 0
    total_original_rows = 0
    total_resampled_rows = 0
    
    for filename, success, original_rows, resampled_rows, error in results:
        if success:
            successful_count += 1
            total_original_rows += original_rows
            total_resampled_rows += resampled_rows
        else:
            failed_count += 1
            if error:
                print(f"❌ {filename}: {error}")
    
    print("\nResampling finished:")
    print(f"  Successfully processed files: {successful_count}")
    print(f"  Failed files: {failed_count}")
    print(f"  Total original rows: {total_original_rows:,}")
    print(f"  Total resampled rows: {total_resampled_rows:,}")
    if total_original_rows > 0:
        print(f"  Compression ratio: {((total_original_rows-total_resampled_rows)/total_original_rows*100):.1f}%")
    print("  Interval: 30 seconds")
    print("  Fixed start timestamp: 1625133601000 (ms)")


def process_single_anomaly_sample_all_files(args: tuple) -> tuple:
    """
    Process one anomaly sample by extracting data within its fault window from all files.

    As long as at least one file has data in the window, keep this sample and return per-file data.
    
    Args:
        args: (sample_idx, sample_row, metric_files_dict)
    
    Returns:
        tuple: (sample_idx, success, error_msg, all_files_data_dict)
    """
    try:
        sample_idx, sample_row, metric_files_dict = args
        
        # Label fields
        service = sample_row['service']
        instance = sample_row['instance']
        st_time_str = sample_row['st_time']
        anomaly_type = sample_row['anomaly_type']
        
        # Convert time string to timestamp (ms).
        dt = datetime.strptime(st_time_str, '%Y-%m-%d %H:%M:%S.%f')
        start_timestamp_ms = int(dt.timestamp() * 1000)
        end_timestamp_ms = start_timestamp_ms + 600 * 1000  # +600 seconds
        
        # Store data and timestamps across files.
        all_files_data = {}
        all_actual_timestamps = set()
        has_any_data = False
        
        # Pass 1: scan all files and collect timestamps.
        for file_key, metric_file_path in metric_files_dict.items():
            try:
                # Read file
                df = pd.read_csv(metric_file_path)
                df.columns = df.columns.str.strip()
                
                # Get metric columns
                metric_columns = [col for col in df.columns if col != 'timestamp']
                
                # Extract window data
                window_data = df[(df['timestamp'] >= start_timestamp_ms) & 
                                (df['timestamp'] <= end_timestamp_ms)]
                
                if not window_data.empty:
                    has_any_data = True
                    actual_timestamps = sorted(window_data['timestamp'].unique())
                    all_actual_timestamps.update(actual_timestamps)
                    all_files_data[file_key] = {'metric_columns': metric_columns, 'window_data': window_data}
                else:
                    all_files_data[file_key] = {'metric_columns': metric_columns, 'window_data': pd.DataFrame()}
                    
            except Exception as e:
                # If reading fails, fall back to an empty structure with default columns.
                default_columns = ['docker_memory_fail_count', 'docker_network_out_dropped', 
                                 'docker_diskio_write_service_time', 'docker_diskio_write_bytes',
                                 'docker_diskio_read_service_time', 'docker_network_in_bytes',
                                 'docker_network_out_bytes', 'docker_diskio_read_bytes',
                                 'docker_network_in_errors', 'docker_network_out_errors',
                                 'docker_cpu_total_norm_pct', 'docker_memory_usage_pct']
                all_files_data[file_key] = {'metric_columns': default_columns, 'window_data': pd.DataFrame()}
        
        # If no file has any data in the window, skip this sample.
        if not has_any_data:
            return (sample_idx, False, "no data found in the time window across all files", None)
        
        # Build a unified timestamp sequence.
        if not all_actual_timestamps:
            return (sample_idx, False, "no valid timestamps found", None)
        
        # Compute an aligned start timestamp based on the earliest observed timestamp.
        interval_ms = 30 * 1000
        reference_ts = min(all_actual_timestamps)
        
        # If the earliest timestamp is >= 30 seconds after the fault start time, shift back to align.
        if (reference_ts - start_timestamp_ms) >= interval_ms:
            steps_back = (reference_ts - start_timestamp_ms) // interval_ms
            ideal_first_timestamp = reference_ts - steps_back * interval_ms
        else:
            ideal_first_timestamp = reference_ts
        
        # Generate 20 consecutive timestamps (30s interval).
        expected_timestamps = [ideal_first_timestamp + i * interval_ms for i in range(20)]
        
        # Pass 2: generate complete (20-step) data for each file.
        result_data = {}
        
        for file_key, file_data in all_files_data.items():
            metric_columns = file_data['metric_columns']
            window_data = file_data['window_data']
            
            # Build complete rows for this file.
            complete_data = []
            
            for expected_ts in expected_timestamps:
                # Find data for this timestamp.
                if not window_data.empty:
                    exact_match = window_data[window_data['timestamp'] == expected_ts]
                else:
                    exact_match = pd.DataFrame()
                
                if not exact_match.empty:
                    # Use the exact match (metric columns only).
                    row_data = exact_match.iloc[0][['timestamp'] + metric_columns].to_dict()
                else:
                    # No match: create an empty row.
                    row_data = {'timestamp': float(expected_ts)}
                    # Set all metric columns to None.
                    for col in metric_columns:
                        row_data[col] = None
                
                complete_data.append(row_data)
            
            # Convert to DataFrame.
            complete_df = pd.DataFrame(complete_data)
            
            # Ensure timestamp is the first column.
            cols = ['timestamp'] + [col for col in complete_df.columns if col != 'timestamp']
            complete_df = complete_df[cols]
            
            result_data[file_key] = complete_df
        
        return (sample_idx, True, None, result_data)
        
    except Exception as e:
        return (sample_idx, False, str(e), None)


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
    
    # Convert to timestamps; each window lasts 600 seconds from start.
    anomaly_periods = []
    for _, row in label_df.iterrows():
        # Convert start time string to timestamp (ms).
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000  # start + 600 seconds
        data_type = row.get('data_type', 'unknown')  # default: unknown
        anomaly_periods.append((st_time, ed_time, data_type))
    
    print(f"Loaded {len(anomaly_periods)} anomaly periods")
    return anomaly_periods


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


def extract_anomaly_metric_data(metric_dir, anomaly_periods):
    """
    Extract metric rows within anomaly windows for all files under processed_data/metric.
    Overwrite the original files in place.
    
    Args:
        metric_dir (str): Metric directory.
        anomaly_periods (list): Anomaly periods.
    """
    print("=== Start extracting anomaly metric data (overwrite originals) ===")
    
    # List metric files.
    metric_files = [f for f in os.listdir(metric_dir) if f.endswith('.csv')]
    
    # Process each file
    for metric_file in metric_files:
        print(f"\nProcessing file: {metric_file}")
        metric_file_path = os.path.join(metric_dir, metric_file)
        
        # Read metric file.
        metric_df = pd.read_csv(metric_file_path)
        original_count = len(metric_df)
        print(f"  Rows read: {original_count:,}")
        
        # Check for timestamp column.
        if 'timestamp' not in metric_df.columns:
            print(f"  ⚠️  Skipping {metric_file}: missing timestamp column")
            continue
        
        # Create selection mask for anomaly windows.
        anomaly_mask = create_selection_mask(metric_df['timestamp'], anomaly_periods)
        
        # Extract anomaly rows.
        anomaly_data = metric_df[anomaly_mask].copy()
        anomaly_count = len(anomaly_data)
        
        # Overwrite original file.
        anomaly_data.to_csv(metric_file_path, index=False)
        
        print(f"  Extracted anomaly rows: {anomaly_count:,} ({anomaly_count/original_count:.2%})")
        print(f"  Overwrote original file: {metric_file_path}")
    
    print("\n=== Anomaly metric extraction finished ===")


def extract_anomaly_samples(num_processes: int = None) -> None:
    """
    Extract anomaly samples' metric data in parallel.

    For each sample in the label file, extract data within the 600-second window
    starting from the fault start timestamp.
    
    Args:
        num_processes: Number of processes to use (default: CPU cores).
    
    Returns:
        None
    """
    # Paths
    label_file = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'label_gaia.csv')
    processed_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anomaly_samples')
        
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    
    # Read label file
    print("📖 Reading label file...")
    labels_df = pd.read_csv(label_file)
    print(f"✅ Loaded {len(labels_df)} anomaly samples")
    
    # Build metric file mapping
    print("🔍 Building metric file mapping...")
    metric_files_dict = {}
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.csv'):
            # Extract service instance name from filename
            # Example: dbservice1_0.0.0.4_2021-07-01_2021-07-31.csv -> dbservice1
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                instance_name = parts[0]  # use service instance name as key, e.g., dbservice1
                metric_files_dict[instance_name] = os.path.join(processed_data_dir, filename)
    
    print(f"✅ Found {len(metric_files_dict)} metric files:")
    for instance, path in metric_files_dict.items():
        print(f"   {instance}: {os.path.basename(path)}")
    
    # Prepare multiprocessing arguments
    process_args = []
    for idx, row in labels_df.iterrows():
        process_args.append((idx, row, metric_files_dict))
    
    # Determine process count
    if num_processes is None:
        num_processes = min(cpu_count(), len(process_args))
    
    print(f"\n🚀 Extracting {len(process_args)} anomaly samples using {num_processes} processes...")
    
    # Run multiprocessing pool
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_anomaly_sample, process_args),
            total=len(process_args),
            desc="Extracting anomaly samples"
        ))
    
    # Collect results and save
    successful_samples = []
    failed_count = 0
    
    for sample_idx, success, error_msg, extracted_data in results:
        if success and extracted_data is not None:
            successful_samples.append(extracted_data)
        else:
            failed_count += 1
            if error_msg:
                print(f"❌ Sample {sample_idx}: {error_msg}")
    
    # Merge all extracted data and save
    if successful_samples:
        print("\n💾 Saving extracted data...")
        all_samples_df = pd.concat(successful_samples, ignore_index=True)
        
        # Save full dataset
        all_samples_path = os.path.join(output_dir, 'all_anomaly_samples.csv')
        all_samples_df.to_csv(all_samples_path, index=False)
        print(f"✅ Full dataset saved: {all_samples_path}")
        
        # Save per service
        for service in all_samples_df['service'].unique():
            service_data = all_samples_df[all_samples_df['service'] == service]
            service_path = os.path.join(output_dir, f'{service}_anomaly_samples.csv')
            service_data.to_csv(service_path, index=False)
            print(f"✅ {service} saved: {service_path} ({len(service_data)} rows)")
        
        # Summary
        print("\n📊 Extraction summary:")
        print(f"  Successful samples: {len(successful_samples)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Total data points: {len(all_samples_df):,} (20 timestamps per sample)")
        print("  Window length: 600 seconds (30-second interval)")
        print(f"  Covered services: {', '.join(all_samples_df['service'].unique())}")
        print(f"  Num columns: {len(all_samples_df.columns)}")
        
        # Missing value statistics
        total_metric_values = 0
        missing_metric_values = 0
        metric_columns = [col for col in all_samples_df.columns 
                         if col not in ['timestamp', 'sample_idx', 'service', 'instance', 'anomaly_type', 'start_timestamp']]
        
        for col in metric_columns:
            total_values = len(all_samples_df)
            # Count missing values (None/NaN and empty strings)
            missing_values = all_samples_df[col].isna().sum() + (all_samples_df[col] == '').sum()
            total_metric_values += total_values
            missing_metric_values += missing_values
        
        completion_rate = ((total_metric_values - missing_metric_values) / total_metric_values * 100) if total_metric_values > 0 else 0
        print(f"  Completion rate: {completion_rate:.1f}% ({total_metric_values - missing_metric_values:,}/{total_metric_values:,})")
    else:
        print("❌ No data samples were extracted successfully")


def extract_anomaly_samples_all_files(num_processes: int = None) -> None:
    """
    Extract anomaly samples' metric data from all files in parallel.

    As long as at least one file has data, keep the sample and write per-file outputs.
    
    Args:
        num_processes: Number of processes to use (default: CPU cores).
    
    Returns:
        None
    """
    # Paths
    label_file = os.path.join(_project_root, 'data', 'raw_data', 'gaia', 'label_gaia.csv')
    processed_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    output_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'anomaly_metric')
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    
    # Read label file
    print("📖 Reading label file...")
    labels_df = pd.read_csv(label_file)
    print(f"✅ Loaded {len(labels_df)} anomaly samples")
    
    # Build mapping for all metric files
    print("🔍 Building mapping for all metric files...")
    metric_files_dict = {}
    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.csv'):
            # Use full filename (without extension) as key for per-file saving.
            file_key = filename.replace('.csv', '')
            metric_files_dict[file_key] = os.path.join(processed_data_dir, filename)
    
    # Prepare multiprocessing arguments
    process_args = []
    for idx, row in labels_df.iterrows():
        process_args.append((idx, row, metric_files_dict))
    
    # Determine process count
    if num_processes is None:
        num_processes = min(cpu_count(), len(process_args))
    
    print(f"\n🚀 Extracting {len(process_args)} anomaly samples from all files using {num_processes} processes...")
    
    # Run multiprocessing pool
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_anomaly_sample_all_files, process_args),
            total=len(process_args),
            desc="Extracting anomaly samples"
        ))
    
    # Collect results and save per file
    successful_samples = []
    failed_count = 0
    
    # Initialize list per file
    file_data_dict = {file_key: [] for file_key in metric_files_dict.keys()}
    
    for sample_idx, success, error_msg, all_files_data in results:
        if success and all_files_data is not None:
            successful_samples.append(sample_idx)
            # Append each file's data to its corresponding list.
            for file_key, file_df in all_files_data.items():
                file_data_dict[file_key].append(file_df)
        else:
            failed_count += 1
            if error_msg:
                print(f"❌ Sample {sample_idx}: {error_msg}")
    
    # Merge and save each file's extracted data
    if successful_samples:
        print("\n💾 Saving extracted data to anomaly_metric...")
        
        total_records = 0
        for file_key, data_list in file_data_dict.items():
            if data_list:  # if this file has data
                # Merge all sample data for this file
                file_combined_df = pd.concat(data_list, ignore_index=True)
                
                # Save to output directory
                output_filename = f"{file_key}_anomaly_samples.csv"
                output_path = os.path.join(output_dir, output_filename)
                file_combined_df.to_csv(output_path, index=False)
                
                total_records += len(file_combined_df)
                print(f"✅ {file_key}: saved {len(file_combined_df)} rows")
            else:
                print(f"⚠️  {file_key}: no data")
        
        # Summary
        print("\n📊 Extraction summary:")
        print(f"  Successful samples: {len(successful_samples)}")
        print(f"  Failed samples: {failed_count}")
        print(f"  Total rows: {total_records:,}")
        print(f"  Files written: {len([f for f in file_data_dict.values() if f])}")
        print("  Timestamps per sample: 20 (30-second interval)")
        print("  Window length: 600 seconds")
        print(f"  Output dir: {output_dir}")
        
        # Compute completion rate (based on the first file that has data).
        sample_file_data = None
        for data_list in file_data_dict.values():
            if data_list:
                sample_file_data = pd.concat(data_list, ignore_index=True)
                break
        
        if sample_file_data is not None:
            metric_columns = [col for col in sample_file_data.columns 
                             if col not in ['timestamp', 'sample_idx', 'service', 'instance', 
                                           'anomaly_type', 'start_timestamp', 'file_source']]
            
            total_metric_values = len(sample_file_data) * len(metric_columns)
            missing_metric_values = 0
            for col in metric_columns:
                missing_values = sample_file_data[col].isna().sum()
                missing_metric_values += missing_values
            
            completion_rate = ((total_metric_values - missing_metric_values) / total_metric_values * 100) if total_metric_values > 0 else 0
            print(f"  Completion rate: {completion_rate:.1f}%")
    else:
        print("❌ No data samples were extracted successfully")


def remove_empty_samples_from_processed_data2() -> None:
    """
    Remove samples that have no data across all files in anomaly_metric.

    A sample is removed only if *all* metric values are missing across *all* files.
    
    Returns:
        None
    """
    processed_data2_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'anomaly_metric')
    
    if not os.path.exists(processed_data2_dir):
        print(f"❌ anomaly_metric directory does not exist: {processed_data2_dir}")
        return
    
    # List CSV files
    csv_files = [f for f in os.listdir(processed_data2_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in anomaly_metric directory")
        return
        
    print(f"🔍 Cleaning empty samples across {len(csv_files)} files...")
    
    # Step 1: read files and determine sample count
    all_files_data = {}
    sample_count = 0
    
    for filename in csv_files:
        file_path = os.path.join(processed_data2_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"⚠️  {filename}: file is empty, skipping")
                continue
                
            all_files_data[filename] = df
            
            # Determine sample count (all files should have the same sample count).
            current_sample_count = len(df) // 20 if len(df) % 20 == 0 else 0
            if sample_count == 0:
                sample_count = current_sample_count
            elif sample_count != current_sample_count:
                print(f"⚠️  {filename}: inconsistent sample count ({current_sample_count} vs {sample_count})")
                
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue
    
    if not all_files_data:
        print("❌ Failed to read any files")
        return
    
    print(f"📊 Detected {sample_count} samples. Starting cross-file analysis...")
    
    # Step 2: keep samples that have any non-null value in any file
    samples_to_keep = []  # sample indices to keep
    
    for sample_idx in range(sample_count):
        has_any_data_across_files = False
        
        # Check whether this sample has any data in any file.
        for filename, df in all_files_data.items():
            sample_start = sample_idx * 20
            sample_end = min(sample_start + 20, len(df))
            sample_data = df.iloc[sample_start:sample_end]
            
            # Metric columns (exclude timestamp)
            metric_columns = [col for col in df.columns if col != 'timestamp']
            
            # Check whether any metric column has a non-null value.
            for col in metric_columns:
                non_null_values = sample_data[col].dropna()
                if len(non_null_values) > 0:
                    has_any_data_across_files = True
                    break
            
            if has_any_data_across_files:
                break
        
        if has_any_data_across_files:
            samples_to_keep.append(sample_idx)
    
    print(f"📊 Result: {len(samples_to_keep)}/{sample_count} samples have data; removing {sample_count - len(samples_to_keep)} empty samples")
    
    # Step 3: rewrite each file keeping only valid samples
    total_removed_samples = sample_count - len(samples_to_keep)
    
    for filename, df in all_files_data.items():
        try:
            valid_samples = []
            
            for sample_idx in samples_to_keep:
                sample_start = sample_idx * 20
                sample_end = min(sample_start + 20, len(df))
                sample_data = df.iloc[sample_start:sample_end]
                valid_samples.append(sample_data)
            
            if valid_samples:
                # Merge retained samples
                cleaned_df = pd.concat(valid_samples, ignore_index=True)
                
                # Save cleaned data
                file_path = os.path.join(processed_data2_dir, filename)
                cleaned_df.to_csv(file_path, index=False)
                
                final_sample_count = len(cleaned_df) // 20
                removed_count = sample_count - final_sample_count
                print(f"✅ {filename}: {sample_count} -> {final_sample_count} samples (removed {removed_count})")
            else:
                # All samples removed: create an empty file
                empty_df = pd.DataFrame(columns=df.columns)
                file_path = os.path.join(processed_data2_dir, filename)
                empty_df.to_csv(file_path, index=False)
                print(f"🗑️  {filename}: all samples removed; file cleared")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    print("\n📊 Cleaning summary:")
    print(f"  Original samples: {sample_count}")
    print(f"  Removed empty samples: {total_removed_samples}")
    print(f"  Kept samples: {len(samples_to_keep)}")
    print(f"  Removal ratio: {(total_removed_samples/sample_count*100):.1f}%" if sample_count > 0 else "  Removal ratio: 0%")


def keep_only_complete_samples_from_processed_data2() -> None:
    """
    Keep only samples that are fully complete (no missing metric values) across all files.

    Any sample that has a missing value in any metric column in any file will be removed.
    
    Returns:
        None
    """
    processed_data2_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'anomaly_metric')
    
    if not os.path.exists(processed_data2_dir):
        print(f"❌ anomaly_metric directory does not exist: {processed_data2_dir}")
        return
    
    # List all CSV files
    csv_files = [f for f in os.listdir(processed_data2_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in anomaly_metric directory")
        return
        
    print(f"🔍 Filtering complete samples across {len(csv_files)} files...")
    
    # Step 1: read files and determine sample count
    all_files_data = {}
    sample_count = 0
    
    for filename in csv_files:
        file_path = os.path.join(processed_data2_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"⚠️  {filename}: file is empty, skipping")
                continue
                
            all_files_data[filename] = df
            
            # Determine sample count (all files should have the same sample count).
            current_sample_count = len(df) // 20 if len(df) % 20 == 0 else 0
            if sample_count == 0:
                sample_count = current_sample_count
            elif sample_count != current_sample_count:
                print(f"⚠️  {filename}: inconsistent sample count ({current_sample_count} vs {sample_count})")
                
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue
    
    if not all_files_data:
        print("❌ Failed to read any files")
        return
    
    print(f"📊 Detected {sample_count} samples. Starting cross-file completeness analysis...")
    
    # Step 2: keep samples that have no missing values across all files
    samples_to_keep = []  # sample indices to keep
    
    for sample_idx in range(sample_count):
        is_complete_across_all_files = True
        
        # Check whether the sample has no missing values in any file.
        for filename, df in all_files_data.items():
            sample_start = sample_idx * 20
            sample_end = min(sample_start + 20, len(df))
            sample_data = df.iloc[sample_start:sample_end]
            
            # Metric columns (exclude timestamp)
            metric_columns = [col for col in df.columns if col != 'timestamp']
            
            # Check missing values
            for col in metric_columns:
                null_count = sample_data[col].isna().sum()
                if null_count > 0:
                    # Missing value found: sample is incomplete.
                    is_complete_across_all_files = False
                    break
            
            if not is_complete_across_all_files:
                break
        
        if is_complete_across_all_files:
            samples_to_keep.append(sample_idx)
    
    print(f"📊 Result: {len(samples_to_keep)}/{sample_count} samples are complete; removing {sample_count - len(samples_to_keep)} incomplete samples")
    
    # Step 3: rewrite each file keeping only complete samples
    total_removed_samples = sample_count - len(samples_to_keep)
    
    for filename, df in all_files_data.items():
        try:
            complete_samples = []
            
            for sample_idx in samples_to_keep:
                sample_start = sample_idx * 20
                sample_end = min(sample_start + 20, len(df))
                sample_data = df.iloc[sample_start:sample_end]
                complete_samples.append(sample_data)
            
            if complete_samples:
                # Merge complete samples
                cleaned_df = pd.concat(complete_samples, ignore_index=True)
                
                # Save filtered data
                file_path = os.path.join(processed_data2_dir, filename)
                cleaned_df.to_csv(file_path, index=False)
                
                final_sample_count = len(cleaned_df) // 20
                removed_count = sample_count - final_sample_count
                print(f"✅ {filename}: {sample_count} -> {final_sample_count} complete samples (removed {removed_count})")
            else:
                # No complete samples: create an empty file
                empty_df = pd.DataFrame(columns=df.columns)
                file_path = os.path.join(processed_data2_dir, filename)
                empty_df.to_csv(file_path, index=False)
                print(f"🗑️  {filename}: no complete samples; file cleared")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    print("\n📊 Completeness filtering summary:")
    print(f"  Original samples: {sample_count}")
    print(f"  Removed incomplete samples: {total_removed_samples}")
    print(f"  Kept complete samples: {len(samples_to_keep)}")
    print(f"  Removal ratio: {(total_removed_samples/sample_count*100):.1f}%" if sample_count > 0 else "  Removal ratio: 0%")
    print("  Completion rate: 100.0% (all retained samples have no missing values)")


if __name__ == "__main__":
    label_file = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'label_gaia.csv')
    metric_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    
    # Copy selected metric files into processed_data/metric
    copy_valid_metric_files()

    # Merge date-range files for the same metric
    merge_date_range_files()

    # Resample metrics to 30-second interval
    resample_metrics_30s_interval()

    # Merge per-metric files into one per service instance
    merge_metrics_by_service_instance()

    # Load anomaly periods
    anomaly_periods = load_anomaly_periods(label_file)
    
    # Extract anomaly-window metric rows (overwrite original files)
    extract_anomaly_metric_data(metric_dir, anomaly_periods)
    

    """
    # Sample cleaning (choose one)
    # remove_empty_samples_from_processed_data2()        # lenient: remove samples that are fully empty across all files
    # keep_only_complete_samples_from_processed_data2()  # strict: keep only samples with no missing values across all files
    """