import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))

def process_sn_traces():
    print("=== Start processing SN trace data ===")
    
    raw_data_dir = os.path.join(_project_root, "data", "raw_data", "sn", "data")
    output_dir = os.path.join(_project_root, "data", "processed_data", "sn", "trace")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all experiment folders
    exp_folders = sorted([f for f in os.listdir(raw_data_dir) if f.startswith("SN.") and os.path.isdir(os.path.join(raw_data_dir, f))])
    
    # Buffer all span records: {service_name: [span_records]}
    # If memory becomes an issue, process in batches.
    service_spans = {} 
    
    print(f"Processing traces from {len(exp_folders)} experiment folders...")
    
    for folder in tqdm(exp_folders, desc="Parsing trace files"):
        spans_json_path = os.path.join(raw_data_dir, folder, "spans.json")
        if not os.path.exists(spans_json_path):
            continue
            
        try:
            with open(spans_json_path, 'r') as f:
                traces_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read {spans_json_path}: {e}")
            continue
            
        # traces_data is a list; each element is a trace object.
        # Each trace object contains traceID, spans, processes, etc.
        
        for trace_obj in traces_data:
            trace_id = trace_obj.get('traceID', '')
            processes = trace_obj.get('processes', {})
            spans = trace_obj.get('spans', [])
            
            # Build processID -> serviceName mapping
            pid_to_service = {}
            for pid, p_info in processes.items():
                s_name = p_info.get('serviceName', 'unknown')

                pid_to_service[pid] = s_name
                
            for span in spans:
                # Basic fields
                pid = span.get('processID')
                service_name = pid_to_service.get(pid, 'unknown')
                
                start_time_ts = span.get('startTime')
                duration_ts = span.get('duration')
                
                if start_time_ts is None or duration_ts is None:
                    continue
                
                # Determine status code from tags
                status_code = 200  # default success
                tags = span.get('tags', [])
                error_tag = False
                http_status = None
                
                for tag in tags:
                    key = tag.get('key', '')
                    val = tag.get('value', '')
                    
                    if key == 'error' and val is True:
                        error_tag = True
                    if key == 'http.status_code':
                        try:
                            http_status = int(val)
                        except:
                            pass
                            
                if http_status is not None:
                    status_code = http_status
                elif error_tag:
                    status_code = 500
                    
                # spanID and references (parentID)
                span_id = span.get('spanID', '')
                parent_id = ''
                references = span.get('references', [])
                for ref in references:
                    # Only treat CHILD_OF as a parent-child relation (ignore FOLLOWS_FROM, etc.)
                    if ref.get('refType') == 'CHILD_OF':
                        parent_id = ref.get('spanID', '')
                        break
                # If there is no CHILD_OF reference, parent_id stays empty (root span).

                # Store
                if service_name not in service_spans:
                    service_spans[service_name] = []
                    
                service_spans[service_name].append({
                    'start_time_ts': start_time_ts,
                    'duration': duration_ts,
                    'trace_id': trace_id,
                    'span_id': span_id,
                    'parent_id': parent_id,
                    'status_code': status_code
                })
                
    # Save as CSV
    print("Saving trace CSV files...")
    for service, records in tqdm(service_spans.items(), desc="Saving CSVs"):
        if not records:
            continue
            
        df = pd.DataFrame(records)
        # Sort
        df = df.sort_values('start_time_ts')
        
        save_path = os.path.join(output_dir, f"{service}_trace.csv")
        df.to_csv(save_path, index=False)
        
    print("=== SN trace processing finished ===")

if __name__ == "__main__":
    process_sn_traces()
