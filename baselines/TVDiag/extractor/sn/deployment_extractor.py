#!/usr/bin/env python3
"""
提取 SN 服务依赖图结构（nodes和edges）
- 从trace数据提取服务调用关系
- 从metric文件名提取服务-节点映射关系 (SN: service==node)
- 生成nodes（服务列表）和edges（依赖关系）

支持三种提取模式：
1. Predefined Static模式（默认）：使用预定义的固定边
2. Dynamic模式：每个故障案例单独从Trace提取edges
3. Static模式：所有案例共享从全量Trace提取的全局edges

注意：SN 数据集缺乏 Host 信息，因此默认不包含同节点影响边。
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import glob

_script_dir = os.path.dirname(os.path.abspath(__file__))
_extractor_dir = os.path.dirname(_script_dir)
_baseline_root = os.path.dirname(_extractor_dir)
_project_root = os.path.dirname(os.path.dirname(_baseline_root))
sys.path.append(_project_root)


def extract_nodes_from_metric(metric_dir):
    """
    从metric目录的文件名中提取服务列表 (Nodes)
    """
    print(f"正在从 {metric_dir} 提取节点列表...")
    csv_files = glob.glob(os.path.join(metric_dir, "*_metric.csv"))
    services = []
    for f in csv_files:
        # filename: {service}_metric.csv
        basename = os.path.basename(f)
        service = basename.replace("_metric.csv", "")
        services.append(service)
    
    services = sorted(list(set(services)))
    print(f"发现 {len(services)} 个服务节点: {services}")
    return services

def load_all_trace_data(trace_dir):
    """
    加载所有trace CSV数据并添加parent_name
    """
    print(f"正在从 {trace_dir} 加载所有Trace数据...")
    csv_files = glob.glob(os.path.join(trace_dir, "*_trace.csv"))
    dfs = []
    
    # 建立 service_name 列 (从文件名提取)
    for f in tqdm(csv_files, desc="Loading Trace CSVs"):
        service_name = os.path.basename(f).replace("_trace.csv", "")
        df = pd.read_csv(f)
        df['service_name'] = service_name
        dfs.append(df)
        
    if not dfs:
        raise ValueError("No trace files found.")
        
    full_trace_df = pd.concat(dfs, ignore_index=True)
    print(f"Total trace records: {len(full_trace_df)}")
    
    # 添加 parent_name
    # 逻辑: self join on parent_id == span_id
    # 需要一个只包含 span_id -> service_name 的映射表
    print("构建调用关系 (Resolving parent names)...")
    span_service = full_trace_df[['span_id', 'service_name']].drop_duplicates(subset=['span_id'])
    span_service = span_service.rename(columns={'service_name': 'parent_name'})

    # Left Join
    # 许多 Span 可能没有 parent (root spans)，或者 parent 不在数据集中
    merged_df = full_trace_df.merge(
        span_service,
        left_on='parent_id',
        right_on='span_id',
        how='left',
        suffixes=('', '_parent')
    )
    
    # 过滤掉没有 parent_name 的记录 (无法构建边)
    edges_df = merged_df.dropna(subset=['parent_name'])
    print(f"Found {len(edges_df)} call relations.")
    
    return edges_df

def extract_edges(trace_df, nodes):
    """
    从 trace DataFrame 中提取边 (indices)
    """
    if trace_df.empty:
        return []
        
    # 提取 unique edges: parent_name -> service_name
    unique_calls = trace_df[['parent_name', 'service_name']].drop_duplicates().values.tolist()
    
    edges = []
    for parent, child in unique_calls:
        if parent in nodes and child in nodes:
            src_idx = nodes.index(parent)
            dst_idx = nodes.index(child)
            edges.append([src_idx, dst_idx])
            
    return edges

# 预定义的 SN 边信息
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

# 预定义的 SN 节点顺序 (字母排序)
PREDEFINED_SN_NODES = ['compose-post-service', 'home-timeline-service', 'media-service', 'nginx-web-server', 'post-storage-service', 'social-graph-service', 
                        'text-service', 'unique-id-service', 'url-shorten-service', 'user-mention-service', 'user-service', 'user-timeline-service']

def convert_predefined_edges_to_indices(predefined_edges, nodes):
    """
    将预定义的边转换为索引对
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
    print(f"开始提取 SN 图结构 (Mode: {mode.upper()})")
    print("=" * 60)
    
    sn_processed_data = os.path.join(_project_root, "data", "processed_data", "sn")
    sn_baseline_output = os.path.join(_baseline_root, "data", "sn", "processed_data", "extracted")
    
    label_path = os.path.join(sn_processed_data, "label_sn.csv")
    metric_dir = os.path.join(sn_processed_data, "metric")
    trace_dir = os.path.join(sn_processed_data, "trace")
    output_dir = sn_baseline_output
    
    # 1. 根据模式决定节点获取方式
    if mode == 'predefined_static':
        # 预定义模式：使用预定义的节点
        nodes = PREDEFINED_SN_NODES
        print(f"使用预定义节点 ({len(nodes)} 个): {nodes}")
    else:
        # 数据驱动模式：从 metric 文件夹提取节点
        nodes = extract_nodes_from_metric(metric_dir)
        print(f"从数据提取节点 ({len(nodes)} 个): {nodes}")
    
    # 2. 加载 Trace 数据 (仅在非 predefined_static 模式下需要)
    all_calls_df = pd.DataFrame()
    if mode != 'predefined_static':
        all_calls_df = load_all_trace_data(trace_dir)
    
    # 3. 提取图结构
    nodes_dict = {}
    edges_dict = {}
    
    label_df = pd.read_csv(label_path)
    print(f"处理 {len(label_df)} 个样本...")
    
    if mode == 'predefined_static':
        # 使用预定义的边（节点已在上面设置）
        print("Building Predefined Fixed Graph...")
        global_edges = convert_predefined_edges_to_indices(PREDEFINED_SN_EDGES, nodes)
        print(f"Predefined Edges ({len(global_edges)})")
        
        for _, row in tqdm(label_df.iterrows(), total=len(label_df)):
            sample_id = row['index']
            nodes_dict[sample_id] = nodes
            edges_dict[sample_id] = global_edges
            
    elif mode == 'static':
        # Static: 全局共享 Edges (从 Trace 提取)
        print("Building Static Graph...")
        global_edges = extract_edges(all_calls_df, nodes)
        print(f"Global Edges ({len(global_edges)}): {global_edges}")
        
        for _, row in tqdm(label_df.iterrows(), total=len(label_df)):
            sample_id = row['index']
            
            # 为了兼容性，nodes 列表对每个样本都是一份拷贝
            nodes_dict[sample_id] = nodes
            edges_dict[sample_id] = global_edges
            
    else:
        # Dynamic: 每个样本单独提取
        print("Building Dynamic Graphs...")
        edge_counts = []
        
        for _, row in tqdm(label_df.iterrows(), total=len(label_df)):
            sample_id = row['index']
            
            # 获取时间窗口 (start_time 是 utc 字符串 -> 转 timestamp)
            # 注意：我们的 trace csv 中的 timestamp 是 float seconds (utc)
            st_ts = pd.to_datetime(row['st_time'], utc=True).timestamp()
            ed_ts = st_ts + row['duration'] # 通常 10s
            
            # 使用 Gaia 风格的布尔掩码筛选
            # 确保列名存在
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

    os.makedirs(output_dir, exist_ok=True)
    
    nodes_file = os.path.join(output_dir, 'nodes.json')
    edges_file = os.path.join(output_dir, 'edges.json')
    
    with open(nodes_file, 'w') as f:
        json.dump(nodes_dict, f)
    with open(edges_file, 'w') as f:
        json.dump(edges_dict, f)
    
    print(f"✅ Nodes and Edges saved to {output_dir}")
    print(f"   - nodes.json: {len(nodes_dict)} samples")
    print(f"   - edges.json: {len(edges_dict)} samples")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='predefined_static', choices=['static', 'dynamic', 'predefined_static'])
    args = parser.parse_args()
    
    process_sn_graph(mode=args.mode)
