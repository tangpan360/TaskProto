"""
模板数量获取工具函数
"""
import os
import pandas as pd


def get_log_template_count(dataset):
    """
    动态获取指定数据集的log模板数量
    
    Args:
        dataset (str): 数据集名称 ('sn', 'tt', 'gaia')
        
    Returns:
        int: 模板数量
    """
    # 获取项目根目录 (utils的上级目录)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    
    template_files = {
        'sn': 'sn_templates.csv',
        'tt': 'tt_templates.csv', 
        'gaia': 'gaia_templates.csv'
    }
    
    template_csv_path = os.path.join(
        _project_root, "data", "processed_data", dataset, 
        "drain_models", template_files[dataset]
    )
    
    template_df = pd.read_csv(template_csv_path)
    num_templates = len(template_df)
    # print(f"{dataset.upper()}数据集: {num_templates}个log模板")
    return num_templates
