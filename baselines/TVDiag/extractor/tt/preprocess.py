import os
import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# 模块级路径变量
_script_dir = os.path.dirname(os.path.abspath(__file__))
_extractor_dir = os.path.dirname(_script_dir)
_baseline_root = os.path.dirname(_extractor_dir)
_project_root = os.path.dirname(os.path.dirname(_baseline_root))

# 添加extractor目录到路径
sys.path.append(_extractor_dir)

from utils import io_util
from config import get_window_size

# 数据集配置
dataset = 'tt'
window_size = get_window_size(dataset)

# 动态路径拼接
tt_raw_data = os.path.join(_project_root, 'data', 'raw_data', 'tt')
tt_processed = os.path.join(_baseline_root, 'data', 'tt', 'processed_data')
detector_dir = os.path.join(tt_processed, 'detector')

# 自动创建detector目录
os.makedirs(detector_dir, exist_ok=True)

# 输入文件路径
label_path = os.path.join(_project_root, 'data', 'processed_data', 'tt', 'label_tt.csv')
pre_data_path = os.path.join(tt_processed, 'pre-data.pkl')

labels = pd.read_csv(label_path)
failure_pre_data: dict = io_util.load(pre_data_path)


normal_metrics = {}
normal_traces = defaultdict(list)

# 用于跟踪已处理的pre_data对象（处理SN共享引用情况）
seen_pre_data_ids = set()

for idx, row in tqdm(labels.iterrows(), total=labels.shape[0]):
    if row['data_type'] == 'test':
        continue
    index = row['index']
    chunk = failure_pre_data[index]
    
    # 检查是否已处理过该对象（通过对象ID判断）
    chunk_id = id(chunk['metric'])
    if chunk_id in seen_pre_data_ids:
        continue
    seen_pre_data_ids.add(chunk_id)
    
    for pod, kpi_dic in chunk['metric'].items():
        if pod not in normal_metrics.keys():
            normal_metrics[pod] = defaultdict(list)
        for kpi, kpi_df in kpi_dic.items():
            normal_metrics[pod][kpi].append(kpi_df)
            
    trace_df = chunk['trace']
    trace_df['operation'] = trace_df['url'].str.split('?').str[0]
    trace_gp = trace_df.groupby(['parent_name', 'service_name', 'operation'], dropna=True)
    for (src, dst, op), call_df in trace_gp:
        name = src + '-' + dst + '-' + op
        normal_traces[name].append(call_df)

# 释放大型数据字典
del failure_pre_data

for pod in normal_metrics.keys():
    for kpi, kpi_dfs in normal_metrics[pod].items():
        normal_metrics[pod][kpi] = pd.concat(kpi_dfs)

# 输出文件路径
normal_traces_path = os.path.join(detector_dir, 'normal_traces.pkl')
normal_metrics_path = os.path.join(detector_dir, 'normal_metrics.pkl')

io_util.save(normal_traces_path, normal_traces)
io_util.save(normal_metrics_path, normal_metrics)

# 释放已保存的数据
del normal_traces, normal_metrics

############################################################################

import numpy as np
from sklearn.ensemble import IsolationForest
from extractor.trace_event_extractor import slide_window
from utils import io_util
import time


# 读取已处理的数据
normal_traces_path = os.path.join(detector_dir, 'normal_traces.pkl')
normal_metrics_path = os.path.join(detector_dir, 'normal_metrics.pkl')

normal_traces = io_util.load(normal_traces_path)
normal_metrics = io_util.load(normal_metrics_path)

metric_detectors = {}
for pod in normal_metrics.keys():
    metric_detectors[pod] = {}
    for kpi, dfs in normal_metrics[pod].items():
        metric_detectors[pod][kpi] = [
            normal_metrics[pod][kpi]['value'].mean(), 
            normal_metrics[pod][kpi]['value'].std()
        ]
st = time.time()
trace_detectors = {}
for name, call_dfs in normal_traces.items():
    trace_detectors[name] = {
        'dur_detector': IsolationForest(random_state=0, n_estimators=5),
        '500_detector': IsolationForest(random_state=0, n_estimators=5),
        '400_detector': IsolationForest(random_state=0, n_estimators=5)
    }
    train_ds, train_500_ep, train_400_ep = [], [], []
    for call_df in call_dfs:
        _, durs, err_500_ps, err_400_ps = slide_window(call_df, window_size)
        train_ds.extend(durs)
        train_500_ep.extend(err_500_ps)
        train_400_ep.extend(err_400_ps)
    if len(train_ds) == 0:
        continue
    dur_clf, err_500_clf, err_400_clf = trace_detectors[name]['dur_detector'], trace_detectors[name]['500_detector'], trace_detectors[name]['400_detector']
    dur_clf.fit(np.array(train_ds).reshape(-1,1))
    err_500_clf.fit(np.array(err_500_ps).reshape(-1,1))
    err_400_clf.fit(np.array(err_400_ps).reshape(-1,1))
    trace_detectors[name]['dur_detector']=dur_clf
    trace_detectors[name]['500_detector']=err_500_clf
    trace_detectors[name]['400_detector']=err_400_clf

# 释放训练数据
del normal_traces, normal_metrics

ed = time.time()
# 保存检测器模型
trace_detector_path = os.path.join(detector_dir, 'trace-detector.pkl')
metric_detector_path = os.path.join(detector_dir, 'metric-detector-strict-host.pkl')

io_util.save(trace_detector_path, trace_detectors)
io_util.save(metric_detector_path, metric_detectors)

print(ed-st)