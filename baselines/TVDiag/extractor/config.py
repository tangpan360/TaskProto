"""
数据集配置文件
包含窗口大小、时间单位等数据集特定参数
"""

DATASET_CONFIG = {
    'gaia': {
        'window_size': 30 * 1000,    # Gaia使用毫秒，30秒 = 30000毫秒
        'description': 'Gaia数据集，时间戳单位为毫秒'
    },
    'sn': {
        'window_size': 10,           # SN使用秒，10秒
        'description': 'SN数据集'
    },
    'tt': {
        'window_size': 20,           # TT使用秒，20秒
        'description': 'TT数据集'
    }
}


def get_window_size(dataset_name):
    """
    获取数据集的滑动窗口大小
    
    Args:
        dataset_name: 数据集名称 ('gaia', 'sn', 'tt')
        
    Returns:
        window_size: 窗口大小（单位取决于数据集）
                    - Gaia: 毫秒
                    - SN/TT: 秒
    
    注意:
        训练和测试阶段必须使用相同的窗口大小！
    """
    config = DATASET_CONFIG.get(dataset_name.lower())
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")
    return config['window_size']


def get_all_configs():
    """获取所有数据集的配置信息"""
    return DATASET_CONFIG


if __name__ == '__main__':
    # 测试配置
    print("=" * 60)
    print("数据集窗口大小配置")
    print("=" * 60)
    for dataset in ['gaia', 'sn', 'tt']:
        ws = get_window_size(dataset)
        desc = DATASET_CONFIG[dataset]['description']
        print(f"{dataset.upper():6s}: {ws:8} - {desc}")

