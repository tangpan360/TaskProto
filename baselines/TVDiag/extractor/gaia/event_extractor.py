import os
import sys
from collections import defaultdict
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

# 定义模块级路径变量
_script_dir = os.path.dirname(os.path.abspath(__file__))
_extractor_dir = os.path.dirname(_script_dir)
_baseline_root = os.path.dirname(_extractor_dir)
_project_root = os.path.dirname(os.path.dirname(_baseline_root))

# 添加extractor目录到路径
sys.path.append(_extractor_dir)

from extractor.metric_event_extractor import extract_metric_events
from extractor.trace_event_extractor import extract_trace_events
from extractor.log_event_extractor import extract_log_events
from utils import io_util
from config import get_window_size

# 动态路径拼接
gaia_raw_data = os.path.join(_project_root, 'data', 'raw_data', 'gaia')
gaia_processed = os.path.join(_baseline_root, 'data', 'gaia', 'processed_data')
extracted_dir = os.path.join(gaia_processed, 'extracted')

# 创建统一的提取特征目录
os.makedirs(extracted_dir, exist_ok=True)


# 输入文件路径
post_data_path = os.path.join(gaia_processed, 'post-data-10.pkl')
label_path = os.path.join(gaia_raw_data, 'label_gaia.csv')
metric_detector_path = os.path.join(gaia_processed, 'detector', 'metric-detector-strict-host.pkl')
trace_detector_path = os.path.join(gaia_processed, 'detector', 'trace-detector.pkl')
drain_model_path = os.path.join(gaia_processed, 'drain', 'gaia-drain.pkl')

# 加载数据和模型
data: dict = io_util.load(post_data_path)
# 将第一列设置为索引
label_df = pd.read_csv(label_path, index_col=0)

metric_detectors = io_util.load(metric_detector_path)
trace_detectors = io_util.load(trace_detector_path)

# 从配置文件读取窗口大小
window_size = get_window_size('gaia')
print(f"[Gaia] 使用窗口大小: {window_size}ms")

# 预加载Drain模型（避免循环内重复加载）
miner = io_util.load(drain_model_path)


metric_events_dic = defaultdict(list)
trace_events_dic = defaultdict(list)
log_events_dic = defaultdict(list)
metric_costs, trace_costs, log_costs = [], [], []

for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    chunk = data[idx]
    # extract metric events
    st = time.time()
    metric_events = []
    for pod_host, kpi_dic in chunk['metric'].items():
        kpi_events = extract_metric_events(pod_host, kpi_dic, metric_detectors[pod_host])
        metric_events.extend(kpi_events)
    metric_costs.append(time.time()-st)
    metric_events_dic[idx]=metric_events
    # extract trace events
    st = time.time()
    trace_events = extract_trace_events(chunk['trace'], trace_detectors, window_size)
    trace_events_dic[idx] = trace_events
    trace_costs.append(time.time()-st)
    # extract log events
    st = time.time()
    log_df = chunk['log']
    log_events = extract_log_events(log_df, miner, 0.5)
    log_events_dic[idx] = log_events
    log_costs.append(time.time()-st)

metric_time = np.mean(metric_costs)
trace_time = np.mean(trace_costs)
log_time = np.mean(log_costs)
print(f'the time cost of extract metric events is {metric_time}')
print(f'the time cost of extract trace events is {trace_time}')
print(f'the time cost of extract log events is {log_time}')
#the time cost of extract metric events is 0.18307018280029297
# the time cost of extract trace events is 0.23339865726162023
# the time cost of extract log events is 0.6638196256618483

# 输出提取的特征文件（重命名为复数形式）
logs_path = os.path.join(extracted_dir, 'logs.json')
metrics_path = os.path.join(extracted_dir, 'metrics.json')
traces_path = os.path.join(extracted_dir, 'traces.json')

io_util.save_json(logs_path, log_events_dic)
io_util.save_json(metrics_path, metric_events_dic)
io_util.save_json(traces_path, trace_events_dic)