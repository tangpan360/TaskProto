import os
import sys
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 添加extractor目录到路径以导入utils
_script_dir = os.path.dirname(os.path.abspath(__file__))
_extractor_dir = os.path.dirname(_script_dir)
sys.path.append(_extractor_dir)

from utils import io_util
from utils.time_util import *

# 模块级路径变量
_baseline_root = os.path.dirname(_extractor_dir)
_project_root = os.path.dirname(os.path.dirname(_baseline_root))

# 动态路径拼接
sn_raw_data = os.path.join(_project_root, 'data', 'raw_data', 'sn')
sn_processed = os.path.join(_baseline_root, 'data', 'sn', 'processed_data')
sn_tmp = os.path.join(sn_processed, 'tmp')

os.makedirs(sn_processed, exist_ok=True)
os.makedirs(sn_tmp, exist_ok=True)

def parse_sn_log_timestamp(log_str):
    """解析SN日志时间戳 [2022-Apr-17 10:12:50.490796]"""
    pattern = r"\[(\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]"
    match = re.search(pattern, log_str)
    if not match:
        return None
    
    time_str = match.group(1)
    try:
        dt = pd.to_datetime(time_str, format="%Y-%b-%d %H:%M:%S.%f", utc=True)
        return dt.timestamp()
    except:
        return None

def process_metrics(data_dir):
    """处理指标数据"""
    metrics = {}
    services = ['compose-post-service', 'home-timeline-service', 'media-service', 
                'nginx-web-server', 'post-storage-service', 'social-graph-service', 
                'text-service', 'unique-id-service', 'url-shorten-service',
                'user-mention-service', 'user-service', 'user-timeline-service']
    
    for service in services:
        csv_path = os.path.join(data_dir, "metrics", f"{service}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 关键：所有时间已统一为UTC
            df['timestamp'] = df['timestamp'].astype(int)
            metrics[service] = df
    
    return metrics

def process_logs(data_dir):
    """处理日志数据"""
    logs_path = os.path.join(data_dir, "logs.json")
    all_logs = []
    
    if os.path.exists(logs_path):
        with open(logs_path, 'r') as f:
            logs_dict = json.load(f)
        
        for service, log_list in logs_dict.items():
            for log_msg in log_list:
                ts = parse_sn_log_timestamp(log_msg)
                if ts is not None:
                    # 关键：log不需要偏移，直接UTC解析
                    all_logs.append({
                        'timestamp': ts,
                        'service': service,
                        'message': log_msg
                    })
    
    return pd.DataFrame(all_logs)

def process_traces(data_dir):
    """处理调用链数据"""
    spans_path = os.path.join(data_dir, "spans.json")
    all_traces = []
    
    if os.path.exists(spans_path):
        with open(spans_path, 'r') as f:
            traces_data = json.load(f)
        
        for trace_obj in traces_data:
            trace_id = trace_obj.get('traceID', '')
            processes = trace_obj.get('processes', {})
            spans = trace_obj.get('spans', [])
            
            pid_to_service = {}
            for pid, p_info in processes.items():
                service_name = p_info.get('serviceName', '')
                if service_name:
                    pid_to_service[pid] = service_name
            
            # 建立spanID到service的映射(用于查找parent)
            span_id_to_service = {}
            for span in spans:
                span_id = span.get('spanID', '')
                process_id = span.get('processID', '')
                service_name = pid_to_service.get(process_id, 'unknown')
                if span_id:
                    span_id_to_service[span_id] = service_name
            
            for span in spans:
                process_id = span.get('processID', '')
                service_name = pid_to_service.get(process_id, 'unknown')
                operation_name = span.get('operationName', 'unknown')
                
                start_time_ts = span.get('startTime')
                duration_ts = span.get('duration')
                
                if start_time_ts is None or duration_ts is None:
                    continue
                
                status_code = 200
                tags = span.get('tags', [])
                for tag in tags:
                    if tag.get('key') == 'http.status_code':
                        try:
                            status_code = int(tag.get('value'))
                        except:
                            pass
                        break
                
                parent_name = None
                references = span.get('references', [])
                for ref in references:
                    if ref.get('refType') == 'CHILD_OF':
                        parent_span_id = ref.get('spanID', '')
                        parent_name = span_id_to_service.get(parent_span_id, None)
                        break
                
                all_traces.append({
                    'start_time': start_time_ts,
                    'end_time': start_time_ts + duration_ts,
                    'duration': duration_ts,
                    'service_name': service_name,
                    'parent_name': parent_name,
                    'url': operation_name,
                    'status_code': status_code,
                    'timestamp': start_time_ts
                })
    
    return pd.DataFrame(all_traces)

def extract_data_window(data, start_time, end_time):
    """按时间窗口提取数据"""
    if isinstance(data, pd.DataFrame):
        return data[(data['timestamp'] >= start_time) & (data['timestamp'] < end_time)]
    elif isinstance(data, dict):
        result = {}
        for service, df in data.items():
            windowed_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]
            if not windowed_df.empty:
                result[service] = windowed_df
        return result
    return data

def convert_metric_to_gaia_format(sn_metrics):
    """SN metric格式 → Gaia metric格式"""
    gaia_metrics = {}
    for service, service_df in sn_metrics.items():
        pod_host = f"{service}_host1"
        gaia_metrics[pod_host] = {}
        
        for col in service_df.columns:
            if col != 'timestamp':
                kpi_df = service_df[['timestamp', col]].copy()
                kpi_df = kpi_df.rename(columns={col: 'value'})
                gaia_metrics[pod_host][col] = kpi_df
                
    return gaia_metrics


if __name__ == '__main__':
    print("=== SN Raw Process for TVDiag ===")
    
    # 1. 处理正常数据（用于训练检测器）
    print("Processing normal data...")
    no_fault_dir = os.path.join(sn_raw_data, "no fault")
    normal_experiments = [d for d in os.listdir(no_fault_dir) 
                            if d.startswith("SN.2022") and os.path.isdir(os.path.join(no_fault_dir, d))]
    
    # 合并所有正常数据
    all_normal_metrics = defaultdict(list)
    all_normal_logs = []
    all_normal_traces = []
    
    for exp in normal_experiments:
        exp_dir = os.path.join(no_fault_dir, exp)
        
        # 处理各模态数据
        metrics = process_metrics(exp_dir)
        logs_df = process_logs(exp_dir)
        traces_df = process_traces(exp_dir)
        
        # 合并到总体数据
        for service, df in metrics.items():
            all_normal_metrics[service].append(df)
        
        if not logs_df.empty:
            all_normal_logs.append(logs_df)
        
        if not traces_df.empty:
            all_normal_traces.append(traces_df)
    
    # 合并正常数据
    pre_data = {'normal': {}}
    
    # 合并指标
    normal_metrics = {}
    for service, dfs in all_normal_metrics.items():
        if dfs:
            combined_df = pd.concat(dfs).sort_values('timestamp').drop_duplicates()
            normal_metrics[service] = combined_df
    
    # 合并日志和调用链
    normal_logs_df = pd.concat(all_normal_logs).sort_values('timestamp') if all_normal_logs else pd.DataFrame()
    normal_traces_df = pd.concat(all_normal_traces).sort_values('timestamp') if all_normal_traces else pd.DataFrame()
    
    pre_data['normal'] = {
        'metric': normal_metrics,
        'log': normal_logs_df,
        'trace': normal_traces_df
    }
    
    # 2. 处理故障数据（按标签切片）
    print("Processing fault data...")
    label_path = os.path.join(_project_root, "data", "processed_data", "sn", "label_sn.csv")
    label_df = pd.read_csv(label_path)
    
    fault_dir = os.path.join(sn_raw_data, "data")
    fault_experiments = [d for d in os.listdir(fault_dir) 
                        if d.startswith("SN.2022") and os.path.isdir(os.path.join(fault_dir, d))]
    
    # 读取所有故障实验数据
    fault_data = {}
    for exp in fault_experiments:
        exp_dir = os.path.join(fault_dir, exp)
        fault_data[exp] = {
            'metric': process_metrics(exp_dir),
            'log': process_logs(exp_dir),
            'trace': process_traces(exp_dir)
        }
    
    # 按标签切片
    post_data = {}
    for _, row in tqdm(label_df.iterrows(), total=len(label_df)):
        idx = row['index']
        st_time = row['st_timestamp']  # 标签时间戳已包含偏移
        ed_time = row['ed_timestamp']
        
        # 根据时间找对应的实验
        target_exp = None
        for exp_name, exp_data in fault_data.items():
            # 检查该时间是否在该实验范围内
            if exp_data['metric']:
                sample_service = list(exp_data['metric'].keys())[0]
                exp_df = exp_data['metric'][sample_service]
                if not exp_df.empty:
                    exp_start = exp_df['timestamp'].min()
                    exp_end = exp_df['timestamp'].max()
                    if exp_start <= st_time <= exp_end:
                        target_exp = exp_name
                        break
        
        if target_exp:
            exp_data = fault_data[target_exp]
            
            # 按时间窗口提取数据
            windowed_metrics = extract_data_window(exp_data['metric'], st_time, ed_time)
            windowed_logs = extract_data_window(exp_data['log'], st_time, ed_time)
            windowed_traces = extract_data_window(exp_data['trace'], st_time, ed_time)
            
            post_data[idx] = {
                'metric': windowed_metrics,
                'log': windowed_logs,
                'trace': windowed_traces
            }
    
    # 3. 格式转换为Gaia兼容格式
    print("Converting to Gaia-compatible format...")
    
    # 先转换normal数据
    converted_normal_data = {
        'metric': convert_metric_to_gaia_format(pre_data['normal']['metric']),
        'log': pre_data['normal']['log'],
        'trace': pre_data['normal']['trace']
    }
    
    # 释放原始pre_data释放内存
    del pre_data
    
    # 转换pre_data结构：为每个样本创建相同的正常数据引用
    print("Creating pre_data structure...")
    gaia_pre_data = {}
    for idx in list(post_data.keys()):
        gaia_pre_data[idx] = converted_normal_data
    
    # 分批转换post_data以节省内存
    print("Converting post_data...")
    gaia_post_data = {}
    batch_size = 100
    post_keys = list(post_data.keys())
    
    for i in range(0, len(post_keys), batch_size):
        batch_keys = post_keys[i:i+batch_size]
        for idx in batch_keys:
            sample = post_data[idx]
            gaia_post_data[idx] = {
                'metric': convert_metric_to_gaia_format(sample['metric']),
                'log': sample['log'],
                'trace': sample['trace']
            }
        
        # 释放已处理的原始数据
        for idx in batch_keys:
            del post_data[idx]
        
        print(f"   Converted batch {i//batch_size + 1}/{(len(post_keys)-1)//batch_size + 1}")
    
    # 释放剩余数据
    del post_data
    
    # 4. 保存数据
    print("Saving processed data...")
    pre_data_path = os.path.join(sn_processed, "pre-data.pkl")
    post_data_path = os.path.join(sn_processed, "post-data-10.pkl")
    
    io_util.save(pre_data_path, gaia_pre_data)
    io_util.save(post_data_path, gaia_post_data)
    
    print(f"✅ SN data processed successfully!")
    print(f"   Normal experiments: {len(normal_experiments)}")
    print(f"   Fault samples: {len(gaia_post_data)}")
    print(f"   Pre-data saved: {pre_data_path}")
    print(f"   Post-data saved: {post_data_path}")
    print(f"   Format: Gaia-compatible for direct script reuse")
