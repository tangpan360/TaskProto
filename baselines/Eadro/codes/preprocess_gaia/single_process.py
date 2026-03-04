import numpy as np
from tick.hawkes import HawkesADM4
from tqdm import tqdm

def logHaw(chunk_logs, end_time, event_num, decay=3, ini_intensity=0.2):
    model = HawkesADM4(decay)
    model.fit(chunk_logs, end_time, baseline_start=np.ones(event_num)*ini_intensity)
    return np.array(model.baseline)

from util import read_json, Info
import pandas as pd
import os
import pickle

# 设置相对路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_eadro_root = os.path.dirname(os.path.dirname(_script_dir))
output_path = os.path.join(_eadro_root, 'data')

def deal_logs(intervals, info, idx, name):
    print("*** Dealing with logs...")
    
    df = pd.read_csv(os.path.join(output_path, "parsed_data", name, "logs"+idx+".csv"))
    templates = read_json(os.path.join(output_path, "parsed_data", name, "templates.json"))
    event_num = len(templates)
    
    print("# Real Template Number:", event_num)
    event_num += 1 #0: unseen

    event2id = {temp:idx+1 for idx, temp in enumerate(templates)} #0: unseen
    event2id["Unseen"] = 0
    res = np.zeros((len(intervals), info.node_num, event_num))

    no_log_chunk = 0 
    for chunk_idx, (s, e) in tqdm(enumerate(intervals), total=len(intervals), desc="Processing log chunks"):
        if (chunk_idx+1) % 100 == 0: 
            print("Computing Hawkes of chunk {}/{}".format(chunk_idx+1, len(intervals)))
        try:
            rows = df.loc[(df["timestamp"] >=s) & (df["timestamp"]<=e)]
        except:
            no_log_chunk +=1
            continue

        service_events = rows.groupby("service")
        for service, sgroup in service_events:
            events = sgroup.groupby("events")
            knots = [np.array([0.0]) for _ in range(event_num)]
            for event, egroup in events:
                eid = 0 if event not in event2id else event2id[event]
                tmp = np.array(sorted(egroup["timestamp"].values))-s
                adds = np.array([idx*(1e-5) for idx in range(len(tmp))]) #In case of too many identical numbers
                knots[eid] = tmp+adds
            paras = logHaw(knots, end_time=e+1, event_num=event_num)
            res[chunk_idx, info.service2nid[service], :] = paras
    
    print("# Empty log:", no_log_chunk)   
    with open(os.path.join(output_path, "chunks", name, idx, "logs.pkl"), "wb") as fw:
        pickle.dump(res, fw)
    return res

z_zero_scaler = lambda x: (x-np.mean(x)) / (np.std(x)+1e-8)

def deal_metrics(intervals, info, idx, name, chunk_lenth):
    print("*** Dealing with metrics...")
    metric_num = len(info.metric_names)
    metrics = np.zeros((len(intervals), info.node_num, chunk_lenth, metric_num))
    
    for nid, service in tqdm(enumerate(info.service_names), total=len(info.service_names), desc="Processing metric services"):
        df = pd.read_csv(os.path.join(output_path, "parsed_data", name, "metrics"+idx, service+'.csv'))
        # 先对整个数据集进行归一化
        df[info.metric_names] = df[info.metric_names].apply(z_zero_scaler)
        
        for chunk_idx, (s,e) in enumerate(intervals):
            # 找到 >= s 的第一个timestamp，然后往后取chunk_lenth个数据点
            mask = df['timestamp'] / 1000 >= s
            filtered_df = df[mask].head(chunk_lenth)
            
            if len(filtered_df) >= chunk_lenth:
                # 提取归一化后的metric值
                values = filtered_df[info.metric_names].to_numpy()
                metrics[chunk_idx, nid, :, :] = values
            else:
                print(f"Warning: {service} chunk {chunk_idx} has {len(filtered_df)} rows, expected {chunk_lenth}")
                continue
    
    with open(os.path.join(output_path, "chunks", name, idx, "metrics.pkl"), "wb") as fw:
        pickle.dump(metrics, fw)
    return metrics

def deal_traces(intervals, info, idx, name, chunk_lenth):
    print("*** Dealing with traces...")
    traces = read_json(os.path.join(output_path, "parsed_data", name, "traces"+idx+".json"))
    invocations = []
    slot_duration = 600 // chunk_lenth  # Gaia: 600s / 20slots = 30s per slot
    latency = np.zeros((len(intervals), info.node_num, chunk_lenth, 2))

    for chunk_idx, (s, e) in tqdm(enumerate(intervals), total=len(intervals), desc="Processing trace chunks"):
        invok = {}
        for slot in range(chunk_lenth):
            slot_start = s + slot * slot_duration
            slot_end = s + (slot + 1) * slot_duration
            tmp_node_lat = [[] for _ in range(info.node_num)]
            
            for ts in range(slot_start, slot_end + 1):
                if str(ts) in traces:
                    spans = traces[str(ts)]
                    for service, lat_lst in spans.items():
                        if service not in info.service2nid:
                            continue
                        if service not in invok:
                            invok[service] = 0
                        invok[service] += len(lat_lst)
                        node_id = info.service2nid[service]
                        tmp_node_lat[node_id].extend(lat_lst)
            
            for node_id in range(info.node_num):
                if len(tmp_node_lat[node_id]) > 0:
                    latency[chunk_idx][node_id][slot][0] = np.mean(tmp_node_lat[node_id])
        invocations.append(invok)
    
    for i in range(info.node_num):
        latency[:, i, :, 0] = z_zero_scaler(latency[:, i, :, 0])
    
    chunk_traces = {"invok": invocations, "latency": latency}
    with open(os.path.join(output_path, "chunks", name, idx, "traces.pkl"), "wb") as fw:
        pickle.dump(chunk_traces, fw)
    return chunk_traces
