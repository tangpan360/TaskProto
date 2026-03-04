import os
import json

from util import Info

# 设置相对路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_eadro_root = os.path.dirname(os.path.dirname(_script_dir))
_project_root = os.path.dirname(os.path.dirname(_eadro_root))
root_path = os.path.join(_project_root, 'data', 'raw_data')
output_path = os.path.join(_eadro_root, 'data')

def parse_spans(spans_path, dataset_name, info):
    spans_path = sorted(spans_path)    
    for idx, path in enumerate(spans_path):
        with open(path, 'r') as f:
            data = json.load(f)
        t_s_l = []
        for trace in data:
            processes = trace['processes']
            for span in trace['spans']:
                service = processes[span['processID']]['serviceName']
                service = service + '-' + str(info.service2nid[service])
                startTime = int(span['startTime'])  # 整数秒
                latency = float(span['duration'])   # 浮点秒（保留微秒精度）
                t_s_l.append((startTime, service, latency))
        
        # print(len(t_s_l));exit()
        
        unique_timestamps = sorted(list(set([t for t, _, _ in t_s_l])))
        processed_data = {str(t):{} for t in unique_timestamps}
        for t, s, l in t_s_l:
            processed_data[str(t)][s] = processed_data[str(t)].get(s, []) + [l]
        
        save_path = os.path.join(output_path, f'parsed_data/{dataset_name}/traces{idx}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(processed_data, f)
        
        # # check the timestamp range is aligned with the datetime in the filename
        # min_timestamp = min(unique_timestamps)
        # max_timestamp = max(unique_timestamps)
        # import pandas as pd
        # min_timestamp = pd.to_datetime(min_timestamp, unit='s')
        # max_timestamp = pd.to_datetime(max_timestamp, unit='s')
        # print(path)
        # print(min_timestamp, max_timestamp)
        # print("+---------------------------------+")


if __name__ == '__main__':
    for dataset_name in ['SN', 'TT']:
        
        info = Info('trainticket') if dataset_name == 'TT' else Info('socialnetwork')
        
        if dataset_name == 'SN':
            dataset_path = os.path.join(root_path, 'sn', 'data')
        elif dataset_name == 'TT':
            dataset_path = os.path.join(root_path, 'tt', 'data')
    
        # list all folders in the dataset path
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        spans_path = [os.path.join(dataset_path, f, 'spans.json') for f in folders]
        
        parse_spans(spans_path, dataset_name, info)
