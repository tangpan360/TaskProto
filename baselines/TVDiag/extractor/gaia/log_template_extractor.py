import os
import sys
import pandas as pd
from tqdm import tqdm

# 定义模块级路径变量
_script_dir = os.path.dirname(os.path.abspath(__file__))
_extractor_dir = os.path.dirname(_script_dir)
_baseline_root = os.path.dirname(_extractor_dir)
_project_root = os.path.dirname(os.path.dirname(_baseline_root))

# 添加extractor目录到路径
sys.path.append(_extractor_dir)

from drain.drain_template_extractor import *
from utils import io_util

# 动态路径拼接
gaia_raw_data = os.path.join(_project_root, 'data', 'raw_data', 'gaia')
gaia_processed = os.path.join(_baseline_root, 'data', 'gaia', 'processed_data')
drain_dir = os.path.join(gaia_processed, 'drain')

# 自动创建输出目录
os.makedirs(drain_dir, exist_ok=True)

# 输入文件路径
post_data_path = os.path.join(gaia_processed, 'post-data-10.pkl')
label_path = os.path.join(gaia_raw_data, 'label_gaia.csv')

data: dict = io_util.load(post_data_path)
label_df = pd.read_csv(label_path, index_col=0)

logs = []
for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    if row['data_type'] == 'test':
        continue
    chunk = data[idx]
    logs.extend(chunk['log']['message'].values.tolist())

# Drain模型保存路径
drain_model_path = os.path.join(drain_dir, 'gaia-drain.pkl')

miner = extract_templates(
    log_list=logs, 
    save_pth=drain_model_path)
# miner = io_util.load(drain_model_path)

sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)
template_ids = []
template_counts = []
templates = []

for cluster in sorted_clusters:
    templates.append(cluster.get_template())
    template_ids.append(cluster.cluster_id)
    template_counts.append(cluster.size)

template_df = pd.DataFrame(data={
    'id': template_ids,
    'template': templates,
    'count': template_counts
})

# 模板CSV保存路径
template_csv_path = os.path.join(drain_dir, 'gaia-template.csv')
template_df.to_csv(template_csv_path, index=False)
