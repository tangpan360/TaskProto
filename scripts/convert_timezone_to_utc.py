#!/usr/bin/env python3
"""
Dataset timezone conversion tool (SN and TT).
Normalize all timestamps to UTC.

What will be converted:
1. Folder names: minus 16 hours
2. JSON filenames: minus 16 hours
3. JSON fault/label files: unchanged (already UTC)
4. logs.json: timestamp strings minus 8 hours
5. spans.json: startTime minus 8 hours and converted to seconds; duration converted to seconds
6. metrics: unchanged (already UTC)

Supported datasets: SN (Social Network), TT (Train Ticket)
"""

import json
import shutil
import re
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import time

# Module-level private variables: resolve script/project paths.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

class TimezoneConverter:
    def __init__(self, base_dir: str = None, dataset_type: str = None):
        """
        Initialize the converter.
        
        Args:
            base_dir: Project root directory. If None, auto-detect.
            dataset_type: Dataset type ('sn', 'tt', or None for all)
        """
        # Use provided base_dir or default to project root.
        if base_dir is None:
            base_dir = _project_root
        
        self.base_dir = Path(base_dir)
        self.dataset_type = dataset_type
        self.raw_data_dir = self.base_dir / "data" / "raw_data"
        self.backup_dir = self.base_dir / "data" / "raw_data_backup"
    
    def backup_data(self) -> bool:
        """Create a backup (only datasets to be modified: sn and tt)."""
        print("Checking backup...")
        
        if self.backup_dir.exists():
            # Read backup timestamp info.
            timestamp_file = self.backup_dir / "backup_time.txt"
            if timestamp_file.exists():
                with open(timestamp_file, 'r') as f:
                    backup_info = f.read()
                print("✓ Backup already exists:")
                print(f"  {backup_info.strip()}")
                print("  Skipping backup (keeping existing backup)")
            else:
                print("⚠ Backup directory exists but has no timestamp file")
            return True
        
        print("Creating backup (SN and TT only)...")
        try:
            start_time = time.time()
            
            # Create backup directory.
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Only back up sn and tt (do not back up gaia).
            dataset_types = ['sn', 'tt'] if self.dataset_type is None else [self.dataset_type]
            
            for dt in dataset_types:
                src_dir = self.raw_data_dir / dt
                dst_dir = self.backup_dir / dt
                
                if src_dir.exists():
                    print(f"  Backing up {dt.upper()} dataset...")
                    shutil.copytree(src_dir, dst_dir)
                else:
                    print(f"  ⚠ {dt.upper()} directory does not exist, skipping")
            
            # Write backup timestamp file.
            timestamp_file = self.backup_dir / "backup_time.txt"
            with open(timestamp_file, 'w') as f:
                f.write(f"Backup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source path: {self.raw_data_dir}\n")
                datasets = ', '.join(dataset_types)
                f.write(f"Datasets: {datasets}\n")
                f.write("Note: gaia was not backed up (it is not modified)\n")
            
            elapsed = time.time() - start_time
            print(f"✓ Backup completed! Elapsed: {elapsed:.1f}s")
            return True
        except Exception as e:
            print(f"✗ Backup failed: {e}")
            return False
    
    def convert_folder_name(self, folder_name: str) -> str:
        """Convert folder name (minus 16 hours)."""
        # Parse time: supports SN.xxx or TT.xxx format.
        pattern = r'((?:SN|TT)\.)(\d{4}-\d{2}-\d{2}T\d{6})D(\d{4}-\d{2}-\d{2}T\d{6})'
        match = re.match(pattern, folder_name)
        
        if not match:
            return folder_name
        
        prefix, start_str, end_str = match.groups()
        
        # Convert start time.
        start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H%M%S")
        start_utc = start_dt - timedelta(hours=16)
        
        # Convert end time.
        end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H%M%S")
        end_utc = end_dt - timedelta(hours=16)
        
        new_name = f"{prefix}{start_utc.strftime('%Y-%m-%dT%H%M%S')}D{end_utc.strftime('%Y-%m-%dT%H%M%S')}"
        return new_name
    
    def convert_logs_json(self, file_path: Path) -> int:
        """Convert timestamp strings in logs.json (minus 8 hours).
        
        Supports two formats:
        1) SN: [2022-Apr-17 10:12:50.490796]
        2) TT: 2022-04-17 13:22:01.494
        """
        with open(file_path, 'r') as f:
            logs_data = json.load(f)
        
        # Month mapping for SN format.
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month_map_reverse = {v: k for k, v in month_map.items()}
        
        # Regex patterns for two log formats.
        pattern_sn = r'\[(\d{4})-(\w+)-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)\]'  # SN
        pattern_tt = r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)'    # TT
        
        def convert_sn_format(match):
            """Convert SN format: [2022-Apr-17 10:12:50.490796]."""
            year, month_str, day, hour, minute, second, microsec = match.groups()
            
            dt = datetime(
                int(year), 
                month_map[month_str], 
                int(day), 
                int(hour), 
                int(minute), 
                int(second), 
                int(microsec)
            )
            dt_utc = dt - timedelta(hours=8)
            
            new_month = month_map_reverse[dt_utc.month]
            return f"[{dt_utc.year}-{new_month}-{dt_utc.day:02d} {dt_utc.hour:02d}:{dt_utc.minute:02d}:{dt_utc.second:02d}.{microsec}]"
        
        def convert_tt_format(match):
            """Convert TT format: 2022-04-17 13:22:01.494."""
            year, month, day, hour, minute, second, millisec = match.groups()
            
            dt = datetime(
                int(year), 
                int(month), 
                int(day), 
                int(hour), 
                int(minute), 
                int(second), 
                int(millisec) * 1000  # milliseconds -> microseconds
            )
            dt_utc = dt - timedelta(hours=8)
            
            # Keep TT formatting (numeric date; millisecond precision).
            return f"{dt_utc.year:04d}-{dt_utc.month:02d}-{dt_utc.day:02d} {dt_utc.hour:02d}:{dt_utc.minute:02d}:{dt_utc.second:02d}.{millisec}"
        
        # Convert all logs.
        total_logs = 0
        for service in logs_data:
            converted_logs = []
            for log in logs_data[service]:
                # Try SN format first.
                new_log = re.sub(pattern_sn, convert_sn_format, log)
                # If SN didn't match (log unchanged), try TT format.
                if new_log == log:
                    new_log = re.sub(pattern_tt, convert_tt_format, log)
                converted_logs.append(new_log)
            
            logs_data[service] = converted_logs
            total_logs += len(converted_logs)
        
        # Write back.
        with open(file_path, 'w') as f:
            json.dump(logs_data, f, indent=2)
        
        return total_logs
    
    def convert_spans_json(self, file_path: Path) -> Tuple[int, int]:
        """Convert timestamps in spans.json (minus 8 hours and convert to seconds)."""
        print(f"  Reading {file_path.name}...")
        with open(file_path, 'r') as f:
            spans_data = json.load(f)
        
        total_spans = 0
        total_traces = len(spans_data)
        
        # Convert startTime and duration for each span.
        for trace in spans_data:
            for span in trace.get('spans', []):
                # startTime: minus 8 hours then convert to seconds.
                if 'startTime' in span:
                    # Original value is microseconds: subtract 8 hours (28800000000 us), then convert to seconds.
                    span['startTime'] = (span['startTime'] - 28800000000) / 1000000.0
                
                # duration: only convert unit to seconds (no time shift).
                if 'duration' in span:
                    span['duration'] = span['duration'] / 1000000.0
                
                total_spans += 1
        
        print(f"  Writing {file_path.name}...")
        
        # Write back.
        with open(file_path, 'w') as f:
            json.dump(spans_data, f, indent=2)
        
        return total_traces, total_spans
    
    def convert_dataset(self, dataset_type: str, category: str, dataset_name: str) -> Dict:
        """Convert a single dataset."""
        print("=" * 80)
        print(f"Start converting dataset: [{dataset_type}/{category}] {dataset_name}")
        print("=" * 80)
        
        data_dir = self.raw_data_dir / dataset_type / category
        dataset_dir = data_dir / dataset_name
        
        # Determine prefix from dataset name.
        prefix = dataset_name.split('.')[0]  # SN or TT
        fault_file = data_dir / f"{dataset_name.replace(f'{prefix}.', f'{prefix}.fault-')}.json"
        
        if not dataset_dir.exists():
            print(f"✗ Dataset directory does not exist: {dataset_dir}")
            return {"success": False}
        
        start_time = time.time()
        stats = {
            "success": True,
            "dataset": dataset_name,
            "category": category,
            "logs_count": 0,
            "traces_count": 0,
            "spans_count": 0,
            "elapsed_time": 0
        }
        
        try:
            # Step 1: rename folder and JSON file (simplest operation).
            new_dataset_name = self.convert_folder_name(dataset_name)
            if new_dataset_name != dataset_name:
                print("Step 1: renaming folder and JSON file")
                print(f"  {dataset_name}")
                print(f"  → {new_dataset_name}")
                
                # Rename folder.
                new_dataset_dir = data_dir / new_dataset_name
                dataset_dir.rename(new_dataset_dir)
                dataset_dir = new_dataset_dir  # update path
                
                # Rename JSON file.
                new_fault_file = data_dir / f"{new_dataset_name.replace(f'{prefix}.', f'{prefix}.fault-')}.json"
                if fault_file.exists():
                    fault_file.rename(new_fault_file)
                
                stats["new_name"] = new_dataset_name
                print("  ✓ Rename done")
            
            # Step 2: convert logs.json (medium complexity).
            logs_file = dataset_dir / "logs.json"
            if logs_file.exists():
                print("Step 2: converting logs.json...")
                log_start = time.time()
                stats["logs_count"] = self.convert_logs_json(logs_file)
                print(f"  ✓ Done! Processed {stats['logs_count']} log lines ({time.time()-log_start:.1f}s)")
            
            # Step 3: convert spans.json (most complex operation).
            spans_file = dataset_dir / "spans.json"
            if spans_file.exists():
                print("Step 3: converting spans.json...")
                span_start = time.time()
                stats["traces_count"], stats["spans_count"] = self.convert_spans_json(spans_file)
                print(f"  ✓ Done! Processed {stats['traces_count']} traces, {stats['spans_count']} spans ({time.time()-span_start:.1f}s)")
            
            stats["elapsed_time"] = time.time() - start_time
            print(f"✓ Dataset conversion complete! Total: {stats['elapsed_time']:.1f}s")
            
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
            import traceback
            print(traceback.format_exc())
            stats["success"] = False
            stats["error"] = str(e)
        
        return stats
    
    def convert_all(self) -> List[Dict]:
        """Convert all datasets."""
        print("=" * 80)
        print("Start batch conversion")
        print("=" * 80)
        
        all_stats = []
        
        # Determine which dataset types to process.
        if self.dataset_type:
            dataset_types = [self.dataset_type]
        else:
            # Auto-detect which dataset types exist.
            dataset_types = []
            for dt in ['sn', 'tt']:
                if (self.raw_data_dir / dt).exists():
                    dataset_types.append(dt)
        
        print(f"Dataset types to process: {', '.join(dataset_types).upper()}")
        
        for dataset_type in dataset_types:
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_type.upper()}")
            print('='*80)
            
            type_dir = self.raw_data_dir / dataset_type
            if not type_dir.exists():
                print(f"⚠ Directory does not exist: {type_dir}")
                continue
            
            for category in ["data", "no fault"]:
                category_dir = type_dir / category
                if not category_dir.exists():
                    continue
                
                # Get all dataset folders (SN.xxx or TT.xxx).
                prefix = dataset_type.upper()
                datasets = [d.name for d in category_dir.iterdir() 
                           if d.is_dir() and d.name.startswith(f"{prefix}.")]
                
                if not datasets:
                    continue
                
                print(f"\nFound {len(datasets)} datasets in [{dataset_type}/{category}]")
                
                for dataset_name in sorted(datasets):
                    stats = self.convert_dataset(dataset_type, category, dataset_name)
                    stats["dataset_type"] = dataset_type
                    all_stats.append(stats)
        
        return all_stats


def main():
    """Main entry point."""
    import sys
    
    # Auto-detect project root, process all datasets (sn and tt).
    converter = TimezoneConverter()
    
    print("=" * 80)
    print("Dataset Timezone Conversion Tool (SN + TT)")
    print("=" * 80)
    print(f"Project dir: {converter.base_dir}")
    print(f"Data dir: {converter.raw_data_dir}")
    print(f"Backup dir: {converter.backup_dir}")
    print()
    
    # Step 1: backup
    print("Step 1/2: Create backup")
    if not converter.backup_data():
        print("\nBackup failed, aborting conversion!")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    input("Backup finished. Press Enter to continue, or Ctrl+C to cancel...")
    print()
    
    # Step 2: convert
    print("Step 2/2: Convert data")
    all_stats = converter.convert_all()
    
    # Summary
    print("\n" + "=" * 80)
    print("Conversion summary")
    print("=" * 80)
    
    success_count = sum(1 for s in all_stats if s.get("success"))
    total_logs = sum(s.get("logs_count", 0) for s in all_stats)
    total_spans = sum(s.get("spans_count", 0) for s in all_stats)
    
    # Group by dataset type.
    sn_count = sum(1 for s in all_stats if s.get("dataset_type") == "sn")
    tt_count = sum(1 for s in all_stats if s.get("dataset_type") == "tt")
    
    print(f"Successfully converted: {success_count}/{len(all_stats)} datasets")
    if sn_count > 0:
        print(f"  - SN datasets: {sn_count}")
    if tt_count > 0:
        print(f"  - TT datasets: {tt_count}")
    print(f"Total logs: {total_logs:,}")
    print(f"Total spans: {total_spans:,}")
    
    if success_count == len(all_stats):
        print("\n✓ All datasets converted successfully!")
    else:
        print("\n⚠ Some datasets failed to convert. Please check the logs.")


if __name__ == "__main__":
    main()
