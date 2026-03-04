#!/usr/bin/env python3
"""
Extract service dependency graph structure (nodes and edges).
- Extract service call relations from trace data
- Extract service-to-node mapping from metric filenames
- Produce nodes (service list) and edges (dependencies)

Two extraction modes are supported:

1. Dynamic mode:
   - Extract edges per fault case
   - Based on actual trace calls within the case time window
   - Closer to the original implementation in deployment_extractor.py
   
2. Static mode (default):
   - All cases share a global edge set
   - Build a unified topology from all trace data
   - Useful when the architecture is considered fixed

Optionally include same-node influence edges ("influences"):
   - Default: edges = trace call relations only (pure call topology)
   - With influences: edges = trace call relations + same-node influences

Output file naming:
   - nodes_{mode}_with_influence.json   # with influence edges
   - edges_{mode}_with_influence.json
   - nodes_{mode}_no_influence.json     # without influence edges
   - edges_{mode}_no_influence.json

Usage:
    # Basic usage
    python process_gaia_graph.py                                # default: static + no influences
    python process_gaia_graph.py --mode dynamic                 # dynamic + no influences
    python process_gaia_graph.py --mode static                  # static + no influences
    
    # Include same-node influences
    python process_gaia_graph.py --with-influences              # static + influences (default mode)
    python process_gaia_graph.py --mode dynamic --with-influences # dynamic + influences
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json

# Add project root to path.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.append(_project_root)


def add_parent_name_to_trace(trace_df):
    """
    Add a `parent_name` column via left join.

    Match `parent_id` with `span_id` to obtain the parent service name.
    
    Args:
        trace_df (pd.DataFrame): Raw trace dataframe.
        
    Returns:
        pd.DataFrame: Trace dataframe with a `parent_name` column.
    """
    print("Adding parent_name column...")
    
    # Build a small lookup table: span_id -> parent_name (service_name).
    span_service = trace_df[['span_id', 'service_name']].copy()
    span_service.rename(columns={'service_name': 'parent_name'}, inplace=True)
    
    # Left join: match parent_id to span_id.
    trace_df = trace_df.merge(
        span_service, 
        left_on='parent_id', 
        right_on='span_id', 
        how='left',
        suffixes=('', '_parent')
    )
    
    # Drop redundant column if present.
    if 'span_id_parent' in trace_df.columns:
        trace_df.drop(columns=['span_id_parent'], inplace=True)
    
    # Report ratio of rows with parent_name.
    has_parent = trace_df['parent_name'].notna().sum()
    total = len(trace_df)
    print(f"parent_name added: {has_parent:,}/{total:,} ({has_parent/total:.2%}) rows have a parent service")
    
    return trace_df


def load_trace_data(trace_dir):
    """
    Load all trace CSVs into a single DataFrame, then add `parent_name`.
    
    Args:
        trace_dir (str): Directory containing trace CSV files.
        
    Returns:
        pd.DataFrame: Concatenated trace dataframe (with `parent_name`).
    """
    print(f"Loading trace data from {trace_dir}...")
    
    dfs = []
    trace_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    
    for trace_file in tqdm(trace_files, desc="Reading trace files"):
        trace_path = os.path.join(trace_dir, trace_file)
        df = pd.read_csv(trace_path)
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No trace files found in {trace_dir}")
    
    # Concatenate all trace data.
    trace_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(trace_df):,} trace rows")
    
    # Add parent_name.
    trace_df = add_parent_name_to_trace(trace_df)
    
    return trace_df


def extract_service_node_mapping(metric_dir):
    """
    Extract service-to-node mapping from metric filenames.

    Expected filename pattern:
        {service}_{node}_{metric}_{time}.csv
    
    Args:
        metric_dir (str): Directory containing metric CSV files.
        
    Returns:
        dict: {node_name: [service_names]}
    """
    print(f"Extracting service-to-node mapping from {metric_dir}...")
    
    node2svcs = defaultdict(list)
    
    for filename in os.listdir(metric_dir):
        if not filename.endswith('.csv'):
            continue
            
        splits = filename.split('_')
        if len(splits) < 2:
            continue
            
        svc, host = splits[0], splits[1]
        
        # Filter out system services.
        if svc in ['system', 'redis', 'zookeeper']:
            continue
            
        # Avoid duplicates.
        if svc not in node2svcs[host]:
            node2svcs[host].append(svc)
    
    total_services = sum(len(svcs) for svcs in node2svcs.values())
    print(f"Found {len(node2svcs)} nodes with {total_services} service instances in total")
    
    return dict(node2svcs)


def build_same_node_influences(node2svcs):
    """
    Build bidirectional influence edges between services on the same node.

    For each node, create bidirectional edges for every pair of services.
    
    Args:
        node2svcs (dict): {node_name: [service_names]}
        
    Returns:
        tuple: (sorted_service_list, same_node_influence_edges)
    """
    svcs = []
    influences = []
    
    for node, pods in node2svcs.items():
        svcs.extend(pods)
        
        # Create bidirectional edges among services on the same node.
        for i in range(len(pods)):
            for j in range(i + 1, len(pods)):
                influences.append([pods[i], pods[j]])
                influences.append([pods[j], pods[i]])
    
    # Deduplicate and sort services.
    svcs = sorted(list(set(svcs)))
    
    print(f"Built {len(influences)} same-node influence edges")
    
    return svcs, influences


def extract_trace_call_relations(trace_df):
    """
    Extract service call relations from trace data.

    Each call edge is parent_name -> service_name.
    
    Args:
        trace_df (pd.DataFrame): Trace dataframe.
        
    Returns:
        list: Call relations in the form [[service_name, parent_name], ...]
    """
    # Select rows with a parent service.
    edge_columns = ['service_name', 'parent_name']
    calls = trace_df.dropna(subset=['parent_name']).drop_duplicates(
        subset=edge_columns
    )[edge_columns].values.tolist()
    return calls


def build_graph_structure(svcs, influences, calls, include_influences=True):
    """
    Build graph structure by merging edges and converting service names to indices.
    
    Args:
        svcs (list): Sorted list of services (node names).
        influences (list): Same-node influence edges.
        calls (list): Trace call edges.
        include_influences (bool): Whether to include same-node influences (default: True).
        
    Returns:
        tuple: (nodes, edges) where edges are pairs of indices [src, dst].
    """
    # Merge edges depending on include_influences.
    if include_influences:
        all_edges = calls + influences
    else:
        all_edges = calls
    
    # Deduplicate edges.
    if len(all_edges) > 0:
        all_edges = pd.DataFrame(all_edges).drop_duplicates().reset_index(drop=True).values.tolist()
    else:
        all_edges = []

    # Convert to index-based edges.
    edges = []
    for edge in all_edges:
        # edge[0] is service_name (callee), edge[1] is parent_name (caller).
        # Direction: parent -> service.
        source, target = edge[1], edge[0]
        
        # Ensure both endpoints exist in the service list.
        if source not in svcs or target not in svcs:
            continue
            
        source_idx = svcs.index(source)
        target_idx = svcs.index(target)
        edges.append([source_idx, target_idx])
    
    return svcs, edges


def extract_graph_for_all_cases(label_file, trace_dir, metric_dir, output_dir, 
                                mode='dynamic', include_influences=True):
    """
    Extract graph structure (nodes and edges) for all fault cases.
    
    Args:
        label_file (str): Path to label CSV.
        trace_dir (str): Trace directory.
        metric_dir (str): Metric directory.
        output_dir (str): Output directory.
        mode (str): Extraction mode.
            - 'dynamic': extract edges per case (case-specific topology)
            - 'static': use a single global edge set for all cases
        include_influences (bool): Whether to include same-node influence edges.
            - True: edges include trace calls + same-node influences
            - False: edges include trace calls only
    """
    print("=" * 60)
    print("Extracting graph structure (nodes and edges) for all fault cases")
    print(f"Mode: {mode.upper()}")
    print(f"Include same-node influences: {include_influences}")
    print("=" * 60)
    
    if mode not in ['dynamic', 'static']:
        raise ValueError(f"Unsupported mode: {mode}. Use 'dynamic' or 'static'.")
    
    # 1) Load labels
    print(f"\n1. Loading labels: {label_file}")
    label_df = pd.read_csv(label_file, index_col=0)
    print(f"   Total fault cases: {len(label_df)}")
    
    # 2) Extract service-to-node mapping
    print("\n2. Extracting service-to-node mapping")
    node2svcs = extract_service_node_mapping(metric_dir)
    
    # 3) Build same-node influences
    print("\n3. Building same-node influence edges")
    svcs, influences = build_same_node_influences(node2svcs)
    
    nodes_dict = {}
    edges_dict = {}
    
    if mode == 'static':
        # Static mode: all cases share global edges
        print("\n4. [STATIC] Loading all trace data")
        trace_df = load_trace_data(trace_dir)
        
        print("\n5. [STATIC] Extracting global trace call relations")
        calls = extract_trace_call_relations(trace_df)
        
        print("\n6. [STATIC] Building global graph structure")
        nodes, edges = build_graph_structure(svcs, influences, calls, include_influences)
        
        print("\n7. [STATIC] Assigning the same graph structure to all cases")
        for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Processing cases"):
            nodes_dict[idx] = nodes
            edges_dict[idx] = edges
        
        # Summary stats
        unique_edges = len(edges)
        print(f"\n   Global edges: {unique_edges}")
        
    else:  # mode == 'dynamic'
        # Dynamic mode: extract edges per case
        # Optimization: load all traces into memory once and filter per case.
        print("\n4. [DYNAMIC] Loading all trace data into memory")
        all_trace_df = load_trace_data(trace_dir)
        print(f"   Loaded {len(all_trace_df):,} trace rows in total")
        
        print("\n5. [DYNAMIC] Extracting edges per case")
        edge_stats = []  # edges per case
        
        for idx, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Processing cases"):
            # Get the time window for this case.
            st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
            ed_time = st_time + 600 * 1000  # 600-second window
            
            # Filter trace data for this case from memory.
            if 'start_time_ts' in all_trace_df.columns:
                mask = (all_trace_df['start_time_ts'] >= st_time) & (all_trace_df['start_time_ts'] <= ed_time)
                case_trace_df = all_trace_df[mask]
            else:
                case_trace_df = pd.DataFrame()
            
            if len(case_trace_df) == 0:
                # No trace data: use empty call relations.
                calls = []
            else:
                # Extract call relations for this case.
                calls = extract_trace_call_relations(case_trace_df)
            
            # Build graph for this case.
            nodes, edges = build_graph_structure(svcs, influences, calls, include_influences)
            nodes_dict[idx] = nodes
            edges_dict[idx] = edges
            edge_stats.append(len(edges))
        
        # Summary stats
        import numpy as np
        print("\n   Edge statistics:")
        print(f"     min: {min(edge_stats)}")
        print(f"     max: {max(edge_stats)}")
        print(f"     mean: {np.mean(edge_stats):.1f}")
        print(f"     median: {np.median(edge_stats):.1f}")
    
    # 8) Save results
    print("\n8. Saving results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Filenames include mode and influence flag.
    influence_suffix = "_with_influence" if include_influences else "_no_influence"
    nodes_file = os.path.join(output_dir, f'nodes_{mode}{influence_suffix}.json')
    edges_file = os.path.join(output_dir, f'edges_{mode}{influence_suffix}.json')
    
    with open(nodes_file, 'w') as f:
        json.dump(nodes_dict, f)
    with open(edges_file, 'w') as f:
        json.dump(edges_dict, f)
    
    print(f"   Nodes saved to: {nodes_file}")
    print(f"   Edges saved to: {edges_file}")
    
    # 9) Final summary
    print("\n" + "=" * 60)
    print("Extraction finished. Summary:")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Num cases: {len(nodes_dict)}")
    print(f"Num service nodes: {len(nodes_dict[list(nodes_dict.keys())[0]])}")
    if mode == 'static':
        print(f"Num dependency edges: {unique_edges} (same for all cases)")
        print(f"Graph density: {unique_edges / (len(nodes) * (len(nodes) - 1)):.4f}")
    else:
        print(f"Num dependency edges: mean {np.mean(edge_stats):.1f} (range: {min(edge_stats)}-{max(edge_stats)})")
    print("=" * 60)
    
    return nodes_dict, edges_dict


def main():
    """
    Entry point.
    
    Command-line arguments:
        --mode: extraction mode, 'dynamic' or 'static' (default: static)
        --with-influences: include same-node influence edges (default: False)
        
    Examples:
        python process_gaia_graph.py                                   # default: static + no influences
        python process_gaia_graph.py --mode dynamic                    # dynamic + no influences
        python process_gaia_graph.py --with-influences                 # static + influences
        python process_gaia_graph.py --mode dynamic --with-influences  # dynamic + influences
    """
    import argparse
    
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description='Extract service dependency graph structure (nodes and edges)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode:
  dynamic: dynamic mode
           - Extract edges per fault case
           - Reflects case-specific call relations
           - Edge counts may vary across cases

  static:  static mode (default)
           - All cases share a global edge set
           - Build a unified topology from all trace data
           - More suitable when treating architecture as fixed

Influence edges:
  By default, same-node influence edges ("influences") are not included; only trace call relations are used.
  Use --with-influences to include bidirectional edges among services on the same physical node.
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='static',
        choices=['dynamic', 'static'],
        help="Extraction mode: dynamic (per-case edges) or static (shared global edges)"
    )
    parser.add_argument(
        '--with-influences',
        action='store_true',
        help='Include same-node influence edges (default: disabled)'
    )
    
    args = parser.parse_args()
    
    # Convert to include_influences.
    include_influences = args.with_influences
    
    # Inputs
    label_file = os.path.join(_project_root, "data", "processed_data", "gaia", "label_gaia.csv")
    trace_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "trace")
    metric_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "metric")
    
    # Output
    output_dir = os.path.join(_project_root, "data", "processed_data", "gaia", "graph")
    
    # Run extraction
    print("\nConfig:")
    print(f"  mode: {args.mode}")
    print(f"  include_influences: {include_influences}")
    extract_graph_for_all_cases(label_file, trace_dir, metric_dir, output_dir, 
                                mode=args.mode, include_influences=include_influences)


if __name__ == "__main__":
    main()
