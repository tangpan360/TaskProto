import string
import random
import os
import json
import pandas as pd

# 设置相对路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_eadro_root = os.path.dirname(os.path.dirname(_script_dir))
_project_root = os.path.dirname(os.path.dirname(_eadro_root))
output_path = os.path.join(_eadro_root, 'data')

def load_predefined_labels(name):
    """加载预定义的数据集划分标签，并创建故障类型映射"""
    label_path = os.path.join(_project_root, 'data', 'processed_data',
                               name.lower(), f'label_{name.lower()}.csv')
    df = pd.read_csv(label_path)
    
    # 创建故障类型映射
    fault_types = df['anomaly_type'].unique()
    fault_type_to_idx = {ft: idx for idx, ft in enumerate(sorted(fault_types))}
    
    print(f'# Loaded predefined labels from {label_path}: {len(df)} samples')
    print(f'# Fault type mapping: {fault_type_to_idx}')
    return df, fault_type_to_idx

chunkids = set()
src = string.ascii_letters + string.digits
def get_chunkid():
    while(True):
        chunkid = random.sample(src, 8)
        random.shuffle(chunkid)
        chunkid = ''.join(chunkid)
        #print(chunkid)
        if chunkid not in chunkids:
            chunkids.add(chunkid)
            return chunkid

from util import *
def get_basic(info, label_df, fault_type_to_idx, **kwargs):
    """从预定义的标签数据生成时间窗口和标签"""
    intervals = []
    labels = []
    data_types = []
    fault_types = []
    
    for _, row in label_df.iterrows():
        # 将左闭右开 [st, ed) 转换为闭区间 (st, ed-1)
        # 因为 single_process.py 使用 range(s, e+1) 处理闭区间
        s = int(row['st_timestamp'])
        e = int(row['ed_timestamp']) - 1  # 减1转换为闭区间的右端点
        intervals.append((s, e))
        
        # 服务名映射到节点ID
        service = row['service']
        labels.append(info.service2nid[service])
        
        # 记录数据类型
        data_types.append(row['data_type'])
        
        # 添加故障类型
        fault_type = row['anomaly_type']
        fault_types.append(fault_type_to_idx[fault_type])
    
    print(f'# Using predefined split: {len(intervals)} intervals')
    print(f'# Time range: {intervals[0][0]} to {intervals[-1][-1]}')
    return intervals, labels, data_types, fault_types

import os
import pickle
from collections import defaultdict

from single_process import deal_logs, deal_traces, deal_metrics
def get_chunks(info, name, chunk_lenth, label_df, fault_type_to_idx, idx, **kwargs):
    """从预定义标签生成chunks (单个批次)
    
    Args:
        info: Info对象
        name: 数据集名称 (SN/TT)
        chunk_lenth: 时间窗口长度
        label_df: 标签数据 (当前批次的子集)
        fault_type_to_idx: 故障类型映射字典
        idx: records文件索引 (0, 1, 2, 3...)
    """
    intervals, labels, data_types, fault_types = get_basic(info, label_df, fault_type_to_idx, **kwargs)
    
    aim_dir = os.path.join(output_path, "chunks", name, str(idx))
    os.makedirs(aim_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(aim_dir, "traces.pkl")):
        with open(os.path.join(aim_dir, "traces.pkl"), "rb") as fr: 
            traces = pickle.load(fr)
    else: 
        traces = deal_traces(intervals, info, str(idx), name=name, chunk_lenth=chunk_lenth)
    
    if os.path.exists(os.path.join(aim_dir, "metrics.pkl")):
        with open(os.path.join(aim_dir, "metrics.pkl"), "rb") as fr: 
            metrics = pickle.load(fr)
    else: 
        metrics = deal_metrics(intervals, info, str(idx), name=name, chunk_lenth=chunk_lenth)
    
    if os.path.exists(os.path.join(aim_dir, "logs.pkl")):
        with open(os.path.join(aim_dir, "logs.pkl"), "rb") as fr: 
            logs = pickle.load(fr)
    else: 
        logs = deal_logs(intervals, info, str(idx), name=name)

    print("*** Aligning multi-source data...")
    chunks = defaultdict(dict)
    chunk_data_types = {}
    
    for i in range(len(intervals)):
        chunk_id = get_chunkid()
        chunks[chunk_id]["traces"] = traces["latency"][i]
        chunks[chunk_id]["metrics"] = metrics[i]
        chunks[chunk_id]["logs"] = logs[i]
        chunks[chunk_id]['culprit'] = labels[i]
        chunks[chunk_id]['fault_type'] = fault_types[i]  # 添加故障类型
        chunk_data_types[chunk_id] = data_types[i]

    return chunks, chunk_data_types

import numpy as np
def get_all_chunks(name, chunk_lenth=10, **kwargs):
    """使用预定义标签生成所有chunks (按批次处理)"""
    aim_dir = os.path.join(output_path, "chunks", name)
    os.makedirs(aim_dir, exist_ok=True)
    
    bench = "TrainTicket" if name == "TT" else "SocialNetwork"
    info = Info(bench)
    print('# Node num:', info.node_num)

    # 加载预定义标签和故障类型映射
    label_df, fault_type_to_idx = load_predefined_labels(name)
    
    # 自动适配chunk_lenth为数据duration
    chunk_lenth = int(label_df['duration'].iloc[0])
    
    print("\n\n", "^"*20, "Using predefined split from label CSV", "^"*20)
    
    # 检测有多少个 records 文件
    parsed_data_dir = os.path.join(output_path, "parsed_data", name)
    records_files = sorted([f for f in os.listdir(parsed_data_dir) 
                           if f.startswith('records') and f.endswith('.json')])
    
    print(f"# Found {len(records_files)} records files: {records_files}")
    
    # 读取每个 records 文件的时间范围
    records_ranges = []
    for records_file in records_files:
        records_path = os.path.join(parsed_data_dir, records_file)
        with open(records_path, 'r') as f:
            records_data = json.load(f)
            records_ranges.append((records_data['start'], records_data['end']))
    
    print(f"\n# Records time ranges:")
    for i, (start, end) in enumerate(records_ranges):
        print(f"  records{i}: {start} - {end}")
    
    # 将 label_df 按时间范围分组
    all_chunks = {}
    all_chunk_data_types = {}
    
    for idx, (start, end) in enumerate(records_ranges):
        # 找出在当前时间范围内的样本
        # 注意：label 的 ed_timestamp 是左闭右开，转换后是 ed_timestamp-1
        batch_label_df = label_df[
            (label_df['st_timestamp'] >= start) & 
            (label_df['ed_timestamp'] - 1 <= end)
        ].copy()
        
        if len(batch_label_df) == 0:
            print(f"\n>>> Batch {idx}: No samples in this time range, skipping")
            continue
        
        print(f"\n>>> Processing Batch {idx}: {len(batch_label_df)} samples")
        print(f"    Time range: {batch_label_df['st_timestamp'].min()} - {batch_label_df['ed_timestamp'].max()-1}")
        print(f"    Data types: Train={len(batch_label_df[batch_label_df['data_type']=='train'])}, "
              f"Val={len(batch_label_df[batch_label_df['data_type']=='val'])}, "
              f"Test={len(batch_label_df[batch_label_df['data_type']=='test'])}")
        
        # 处理当前批次
        batch_chunks, batch_chunk_data_types = get_chunks(
            info, name, chunk_lenth, batch_label_df, fault_type_to_idx, idx=idx, **kwargs
        )
        
        print(f"    Generated {len(batch_chunks)} chunks")
        
        # 合并到总结果中
        all_chunks.update(batch_chunks)
        all_chunk_data_types.update(batch_chunk_data_types)
    
    print(f"\n# Total generated chunks: {len(all_chunks)}")
    
    with open(os.path.join(aim_dir, "chunks.pkl"), "wb") as fw:
        pickle.dump(all_chunks, fw)
    
    ############## Update info for metadata ##############
    info.add_info("chunk_lenth", chunk_lenth)
    info.add_info("chunk_num", len(all_chunks))
    info.add_info("edges", info.edges)
    info.add_info("event_num", all_chunks[list(all_chunks.keys())[0]]["logs"].shape[-1])
    info.add_info("fault_type_mapping", fault_type_to_idx)  # 添加故障类型映射
    
    if os.path.exists(os.path.join(aim_dir, "metadata.json")):
        os.remove(os.path.join(aim_dir, "metadata.json"))
    json_pretty_dump(info.metadata, os.path.join(aim_dir, "metadata.json"))

    return all_chunks, all_chunk_data_types

def split_chunks(name, concat=False, **kwargs):
    """按预定义的data_type划分数据集"""
    if os.path.exists(os.path.join(output_path, "chunks", name, "chunks.pkl")):
        with open(os.path.join(output_path, "chunks", name, "chunks.pkl"), "rb") as fr: 
            chunks = pickle.load(fr)
        chunks, chunk_data_types = get_all_chunks(name=name, **kwargs)
    else:
        chunks, chunk_data_types = get_all_chunks(name=name, **kwargs)

    print("\n *** Using predefined train/val/test split...")
    
    # 按 data_type 划分
    train_chunks = {k: v for k, v in chunks.items() 
                   if chunk_data_types.get(k) == 'train'}
    val_chunks = {k: v for k, v in chunks.items() 
                 if chunk_data_types.get(k) == 'val'}
    test_chunks = {k: v for k, v in chunks.items() 
                  if chunk_data_types.get(k) == 'test'}
    
    aim = name[0] if concat else name
    
    # 保存三个文件
    with open(os.path.join(output_path, "chunks", aim, "chunk_train.pkl"), "wb") as fw:
        pickle.dump(train_chunks, fw)
    with open(os.path.join(output_path, "chunks", aim, "chunk_val.pkl"), "wb") as fw:
        pickle.dump(val_chunks, fw)
    with open(os.path.join(output_path, "chunks", aim, "chunk_test.pkl"), "wb") as fw:
        pickle.dump(test_chunks, fw)
    
    # 打印统计
    print(f"\n=== Predefined Split Statistics ===")
    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")
    print(f"Total: {len(chunks)}")
    
    # 统计故障分布
    for split_name, split_chunks in [('Train', train_chunks), 
                                      ('Val', val_chunks), 
                                      ('Test', test_chunks)]:
        fault_count = sum(1 for v in split_chunks.values() if v['culprit'] != -1)
        fault_pct = 100 * fault_count / len(split_chunks) if len(split_chunks) > 0 else 0
        print(f"{split_name}: {fault_count}/{len(split_chunks)} faulty ({fault_pct:.2f}%)")
    
    # 统计每个节点的故障样本数
    label_count = {}
    for _, v in chunks.items(): 
        label = v['culprit']
        if label not in label_count: label_count[label] = 0
        label_count[label] += 1
    
    for label in sorted(list(label_count.keys())):
        if label > -1:
            print('Node {} have {} faulty chunks'.format(label, label_count[label]))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--concat", action="store_true")
parser.add_argument("--delete_all", action="store_true")
parser.add_argument("--delete", action="store_true", help="just remove the final chunks and retain pre-processed data")
parser.add_argument("--chunk_lenth", default=10, type=int)
parser.add_argument("--name", required=True, help="The system name")
params = vars(parser.parse_args())


if "__main__" == __name__:
    aim_dir = os.path.join(output_path, "chunks", params['name'])
    if params['delete_all']:
        _input = input("Do you really want to delete all previous files?! Input yes if you are so confident.\n")
        flag = (_input.lower() == 'yes')
        if flag and os.path.exists(aim_dir) and len(aim_dir)>2:
            import shutil
            shutil.rmtree(aim_dir)
        else:
            print("Thank you for thinking twice!")
            exit()
    if params['delete'] and os.path.exists(os.path.join(aim_dir, "chunks.pkl")):
        os.remove(os.path.join(aim_dir, "chunks.pkl"))
    split_chunks(**params)
