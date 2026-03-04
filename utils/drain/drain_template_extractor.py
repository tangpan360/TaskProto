# -*- coding: utf-8 -*-

"""
    抽取日志模板信息。
"""

import os
import sys
import drain3
import pandas as pd
from tqdm import tqdm
import pickle

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


def init_drain(config_path):
    """
    初始化Drain模板挖掘器
    
    Args:
        config_path (str): 配置文件路径（必须提供）
        
    Returns:
        TemplateMiner: 初始化后的模板挖掘器
    """
    if config_path is None:
        raise ValueError("config_path is required. Please provide the path to drain configuration file.")
    
    config = TemplateMinerConfig()
    print(f"Loading Drain config from: {config_path}")
    config.load(config_path)
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    return template_miner


def extract_templates(log_list: list, save_pth: str, config_path):
    """
    从日志列表中提取模板
    
    Args:
        log_list (list): 日志消息列表
        save_pth (str): 保存模型的路径
        config_path (str): 配置文件路径（必须提供）
        
    Returns:
        TemplateMiner: 训练后的模板挖掘器
    """
    KEEP_TOP_N_TEMPLATE = 1000

    miner = init_drain(config_path=config_path)

    for line in tqdm(log_list, desc="Training Drain"):
        log_txt = line.rstrip()
        miner.add_log_message(log_txt)
    template_count = len(miner.drain.clusters)
    print('The number of templates: {}'.format(template_count))

    template_dict, size_list = {}, []
    for cluster in miner.drain.clusters:
        size_list.append(cluster.size)

    size_list = sorted(size_list, reverse=True)[:KEEP_TOP_N_TEMPLATE]
    min_size = size_list[-1]

    for c in miner.drain.clusters:
        if c.size >= min_size:
            template_dict[c.cluster_id] = c.size

    with open(save_pth, 'wb') as f:
        pickle.dump(miner, f)

    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)
    print('[show templates]')
    for cluster in sorted_clusters:
        print(cluster)

    print("Prefix Tree:")
    miner.drain.print_tree()

    miner.profiler.report(0)

    return miner


def match_template(miner: drain3.TemplateMiner, log_list: list):
    # logger = get_logger("logs_matching")
    IDs = []
    templates = []
    params = []

    for log in tqdm(log_list):
        cluster = miner.match(log)
        
        # logger.debug('match log: {}'.format(log))
        if cluster is None:
            # logger.debug("No match found")
            IDs.append(None)
            templates.append(None)
            
        else:
            template = cluster.get_template()
            param = miner.get_parameter_list(template, log)

            IDs.append(cluster.cluster_id)
            templates.append(template)
            params.append(param)
            # logger.debug(f"Matched template #{cluster.cluster_id}: {template}")

            # params = miner.get_parameter_list(template, log)
            # logger.debug(f"Parameters: {params}")
        # logger.debug("===========================================================================================")

    return IDs, templates, params