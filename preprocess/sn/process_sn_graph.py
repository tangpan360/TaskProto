#!/usr/bin/env python3
"""
Extract SN service dependency graph structure (nodes and edges).
- Extract service call relations from trace data
- Extract service-to-node mapping from metric filenames (SN: service == node)
- Produce nodes (service list) and edges (dependencies)

Three extraction modes are supported:
1. Predefined Static (default): use predefined fixed edges
2. Dynamic: extract edges per fault case from traces
3. Static: all cases share global edges extracted from all traces

Note: The SN dataset lacks host information, so same-node influence edges are not included.
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import glob

# Add project root to path.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)


def extract_nodes_from_metric(metric_dir):
    """
    Extract service list (nodes) from metric filenames.
    """
    print(f"Extracting node list from {metric_dir}...")
    csv_files = glob.glob(os.path.join(metric_dir, "*_metric.csv"))
    services = []
    for f in csv_files:
        # filename: {service}_metric.csv
        basename = os.path.basename(f)
        service = basename.replace("_metric.csv", "")
        services.append(service)
    
    services = sorted(list(set(services)))
    print(f"Found {len(services)} service nodes: {services}")
    return services

def load_all_trace_data(trace_dir):
    """
    Load all trace CSVs and resolve parent_name.
    """
    print(f"Loading all trace data from {trace_dir}...")
    csv_files = glob.glob(os.path.join(trace_dir, "*_trace.csv"))
    dfs = []
    
    # Add service_name column (derived from filename).
    for f in tqdm(csv_files, desc="Loading Trace CSVs"):
        service_name = os.path.basename(f).replace("_trace.csv", "")
        df = pd.read_csv(f)
        df['service_name'] = service_name
        dfs.append(df)
        
    if not dfs:
        raise ValueError("No trace files found.")
        
    full_trace_df = pd.concat(dfs, ignore_index=True)
    print(f"Total trace records: {len(full_trace_df)}")
    
    # Resolve parent_name using a self-join on parent_id == span_id.
    # We need a mapping table span_id -> service_name.
    print("Building call relations (resolving parent names)...")
    span_service = full_trace_df[['span_id', 'service_name']].drop_duplicates(subset=['span_id'])
    span_service = span_service.rename(columns={'service_name': 'parent_name'})
    
    # Left Join
    # Many spans may have no parent (root spans), or the parent may not exist in the dataset.
    merged_df = full_trace_df.merge(
        span_service,
        left_on='parent_id',
        right_on='span_id',
        how='left',
        suffixes=('', '_parent')
    )
    
    # Drop rows without parent_name (cannot form edges).
    edges_df = merged_df.dropna(subset=['parent_name'])
    print(f"Found {len(edges_df)} call relations.")
    
    return edges_df

def extract_edges(trace_df, nodes):
    """
    Extract edges (index pairs) from the trace DataFrame.
    """
    if trace_df.empty:
        return []
        
    # Unique edges: parent_name -> service_name
    unique_calls = trace_df[['parent_name', 'service_name']].drop_duplicates().values.tolist()
    
    edges = []
    for parent, child in unique_calls:
        if parent in nodes and child in nodes:
            src_idx = nodes.index(parent)
            dst_idx = nodes.index(child)
            edges.append([src_idx, dst_idx])
            
    return edges

# Predefined SN edges (service -> list of callees)
PREDEFINED_SN_EDGES = {
    "compose-post-service": ["compose-post-service", "home-timeline-service", "media-service", "post-storage-service", 
                            "text-service", "unique-id-service", "user-service", "user-timeline-service"],
    "home-timeline-service":["home-timeline-service", "post-storage-service", "social-graph-service"],
    "post-storage-service": ["post-storage-service"],
    "social-graph-service": ["social-graph-service", "user-service"],
    "text-service": ["text-service", "url-shorten-service", "user-mention-service"],
    "user-service": ["user-service"],
    "user-timeline-service": ["user-timeline-service"],
    "nginx-web-server": ["compose-post-service", "home-timeline-service", "nginx-web-server", "social-graph-service", "user-service"]
}

# Predefined SN node order (alphabetical)
PREDEFINED_SN_NODES = ['compose-post-service', 'home-timeline-service', 'media-service', 'nginx-web-server', 'post-storage-service', 'social-graph-service', 
                        'text-service', 'unique-id-service', 'url-shorten-service', 'user-mention-service', 'user-service', 'user-timeline-service']

def convert_predefined_edges_to_indices(predefined_edges, nodes):
    """
    Convert predefined edges to index pairs.
    """
    edges = []
    for src_service, dst_services in predefined_edges.items():
        if src_service not in nodes:
            continue
        src_idx = nodes.index(src_service)
        
        for dst_service in dst_services:
            if dst_service in nodes:
                dst_idx = nodes.index(dst_service)
                edges.append([src_idx, dst_idx])
    return edges

def process_sn_graph(mode='dynamic'):
    print("=" * 60)
    print(f"Extracting SN graph structure (Mode: {mode.upper()})")
    print("=" * 60)
    
    # Paths
    label_path = os.path.join(_project_root, "data", "processed_data", "sn", "label_sn.csv")
    metric_dir = os.path.join(_project_root, "data", "processed_data", "sn", "metric")
    trace_dir = os.path.join(_project_root, "data", "processed_data", "sn", "trace")
    output_dir = os.path.join(_project_root, "data", "processed_data", "sn", "graph")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Choose how to get nodes based on mode.
    if mode == 'predefined_static':
        # Predefined mode: use predefined nodes.
        nodes = PREDEFINED_SN_NODES
        print(f"Using predefined nodes ({len(nodes)}): {nodes}")
    else:
        # Data-driven mode: extract nodes from metric files.
        nodes = extract_nodes_from_metric(metric_dir)
        print(f"Extracted nodes from data ({len(nodes)}): {nodes}")
    
    # 2) Load trace data (only needed when not in predefined_static mode).
    all_calls_df = pd.DataFrame()
    if mode != 'predefined_static':
        all_calls_df = load_all_trace_data(trace_dir)
    
    # 3) Build graph structures
    nodes_dict = {}
    edges_dict = {}
    
    label_df = pd.read_csv(label_path)
    print(f"Processing {len(label_df)} samples...")
    
    if mode == 'predefined_static':
        # Use predefined edges (nodes already set above).
        print("Building Predefined Fixed Graph...")
        global_edges = convert_predefined_edges_to_indices(PREDEFINED_SN_EDGES, nodes)
        print(f"Predefined Edges ({len(global_edges)})")
        
        for _, row in tqdm(label_df.iterrows(), total=len(label_df)):
            sample_id = row['index']
            nodes_dict[sample_id] = nodes
            edges_dict[sample_id] = global_edges
            
    elif mode == 'static':
        # Static: global shared edges (extracted from trace)
        print("Building Static Graph...")
        global_edges = extract_edges(all_calls_df, nodes)
        print(f"Global Edges ({len(global_edges)}): {global_edges}")
        
        for _, row in tqdm(label_df.iterrows(), total=len(label_df)):
            sample_id = row['index']
            
            # For compatibility, assign the same node list for each sample.
            nodes_dict[sample_id] = nodes
            edges_dict[sample_id] = global_edges
            
    else:
        # Dynamic: extract per sample.
        print("Building Dynamic Graphs...")
        edge_counts = []
        
        for _, row in tqdm(label_df.iterrows(), total=len(label_df)):
            sample_id = row['index']
            
            # Time window (start_time is a UTC string -> timestamp).
            # Note: trace CSV timestamps are float seconds (UTC).
            st_ts = pd.to_datetime(row['st_time'], utc=True).timestamp()
            ed_ts = st_ts + row['duration']  # typically 10s
            
            # Filter by time window using a boolean mask (Gaia-style).
            # Ensure required columns exist.
            if 'start_time_ts' in all_calls_df.columns:
                mask = (all_calls_df['start_time_ts'] >= st_ts) & (all_calls_df['start_time_ts'] <= ed_ts)
                sample_trace_df = all_calls_df[mask]
            else:
                sample_trace_df = pd.DataFrame()
            
            sample_edges = extract_edges(sample_trace_df, nodes)
            
            nodes_dict[sample_id] = nodes
            edges_dict[sample_id] = sample_edges
            edge_counts.append(len(sample_edges))
            
        print(f"Dynamic Edges Stats: Mean={np.mean(edge_counts):.2f}, Max={np.max(edge_counts)}")

    # 4) Save
    # SN has no host info, so we only save the no_influence version.
    nodes_file = os.path.join(output_dir, f'nodes_{mode}_no_influence.json')
    edges_file = os.path.join(output_dir, f'edges_{mode}_no_influence.json')
    
    with open(nodes_file, 'w') as f:
        json.dump(nodes_dict, f)
    with open(edges_file, 'w') as f:
        json.dump(edges_dict, f)
    
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='predefined_static', choices=['static', 'dynamic', 'predefined_static'])
    args = parser.parse_args()
    
    process_sn_graph(mode=args.mode)
