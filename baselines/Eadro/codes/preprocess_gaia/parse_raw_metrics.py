import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any

# 添加项目根目录到路径，以便导入utils
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
sys.path.append(_project_root)

# 添加Eadro codes路径以便导入Info
eadro_codes_dir = os.path.dirname(_script_dir)
sys.path.append(eadro_codes_dir)
from util import Info


# 全局缓存：预加载的数据
METRIC_DATA_CACHE = {}

# 全局填充统计信息
FILL_STATS = {
    'metric': None,  # {'mean': [12]}
}


def preload_metric_data(info):
    """
    预加载Metric数据到内存
    """
    print("正在加载Metric数据...")
    metric_data_dir = os.path.join(_project_root, 'data', 'processed_data', 'gaia', 'metric')
    for instance_name in tqdm(info.service_names, desc="Loading metric data"):
        metric_file = os.path.join(metric_data_dir, f"{instance_name}_metric.csv")
        if os.path.exists(metric_file):
            df = pd.read_csv(metric_file)
            METRIC_DATA_CACHE[instance_name] = df
    
    print(f"加载完成：{len(METRIC_DATA_CACHE)} 个服务实例")


def load_anomaly_periods(label_file_path):
    """
    加载异常时间段数据（固定600秒窗口）
    """
    label_df = pd.read_csv(label_file_path)
    anomaly_periods = []
    for _, row in label_df.iterrows():
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000
        data_type = row.get('data_type', 'unknown')
        anomaly_periods.append((st_time, ed_time, data_type))
    
    return anomaly_periods


def compute_fill_stats(label_df, info):
    """
    从训练集计算填充统计信息
    """
    train_df = label_df[label_df['data_type'] == 'train']
    print(f"从 {len(train_df)} 个训练样本计算填充统计...")
    
    num_metrics = len(info.metric_names)
    all_metrics = [[] for _ in range(num_metrics)]
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Computing stats"):
        st_time = pd.to_datetime(row['st_time']).timestamp() * 1000
        ed_time = st_time + 600 * 1000
        
        metric = _process_metric_for_sample(st_time, ed_time, info, fill=False)
        
        for i in range(num_metrics):
            vals = metric[:, :, i].flatten()
            valid_vals = vals[~np.isnan(vals)]
            all_metrics[i].extend(valid_vals)
    
    metric_means = np.zeros(num_metrics)
    for i in range(num_metrics):
        if len(all_metrics[i]) > 0:
            metric_means[i] = np.mean(all_metrics[i])
        else:
            metric_means[i] = 0.0
    
    return {'metric': {'mean': metric_means}}


def _process_metric_for_sample(st_time, ed_time, info, fill=True):
    """
    处理单个样本的指标数据
    """
    num_services = len(info.service_names)
    num_metrics = len(info.metric_names)
    metric_data = np.full((num_services, 20, num_metrics), np.nan)
    metric_names = None
    
    for instance_idx, instance_name in enumerate(info.service_names):
        if instance_name not in METRIC_DATA_CACHE:
            continue
            
        df = METRIC_DATA_CACHE[instance_name]
        mask = (df['timestamp'] >= st_time) & (df['timestamp'] <= ed_time)
        sample_data = df[mask].sort_values('timestamp')
        
        if metric_names is None:
            metric_names = [col for col in sample_data.columns if col != 'timestamp']
        
        num_time_steps = min(len(sample_data), 20)
        if num_time_steps > 0:
            metric_data[instance_idx, :num_time_steps, :] = sample_data[metric_names].values[:num_time_steps]
    
    # 填充缺失值
    if fill and FILL_STATS['metric'] is not None:
        stats = FILL_STATS['metric']
        for i in range(num_metrics):
            nan_mask = np.isnan(metric_data[:, :, i])
            if nan_mask.any():
                metric_data[:, :, i][nan_mask] = stats['mean'][i]
    
    return metric_data

def fill_and_export_metrics_directly(info, fill_stats, output_dir):
    """
    直接从原始CSV读取、填充缺失值、导出（方案C）
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在直接处理CSV并导出到: {output_dir}")
    
    for service_name in tqdm(info.service_names, desc="Processing services"):
        if service_name not in METRIC_DATA_CACHE:
            continue
        
        # 读取原始数据（复制以避免修改缓存）
        df = METRIC_DATA_CACHE[service_name].copy()
        
        # 填充缺失值
        for metric_idx, metric_name in enumerate(info.metric_names):
            if metric_name in df.columns:
                nan_mask = df[metric_name].isna()
                if nan_mask.any():
                    df.loc[nan_mask, metric_name] = fill_stats['mean'][metric_idx]
        
        # 保存
        csv_path = os.path.join(output_dir, f"{service_name}.csv")
        df.to_csv(csv_path, index=False)
    
    print(f"导出完成: {len(info.service_names)} 个服务文件")


if __name__ == "__main__":
    # 创建Info对象
    info = Info('gaia')
    
    label_file = os.path.join(_project_root, "data", "processed_data", "gaia", "label_gaia.csv")
    label_df = pd.read_csv(label_file)
    
    # 1. 预加载Metric数据
    preload_metric_data(info)
    
    # 2. 计算或加载填充统计
    stats_file = os.path.join(_project_root, "baselines", "Eadro", "data", "parsed_data", "GAIA", "fill_stats_metric.pkl")
    
    if os.path.exists(stats_file):
        print(f"加载填充统计: {stats_file}")
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
    else:
        print("计算填充统计...")
        stats = compute_fill_stats(label_df, info)
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        print(f"统计信息已保存")
    
    FILL_STATS['metric'] = stats['metric']
    
    # 3. 直接处理原始CSV并导出（方案C）
    output_dir = os.path.join(_project_root, "baselines", "Eadro", "data", "parsed_data", "GAIA", "metrics0")
    fill_and_export_metrics_directly(info, stats['metric'], output_dir)
    
    print(f"\n处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"数据已填充缺失值，归一化将由deal_metrics函数处理")
