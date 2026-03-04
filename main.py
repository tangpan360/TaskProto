import time
import random
import torch
import numpy as np
from config.exp_config import Config
from utils.logger import get_logger
from core.MultiModalDiag import MultiModalDiag
from utils.Result import Result
from process.DatasetProcess import DatasetProcess
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(config: Config, logger):
    processor = DatasetProcess(config, logger)
    train_data, val_data, aug_data, test_data = processor.process()
    return train_data, val_data, aug_data, test_data


def train_and_evaluate(config: Config, log_dir, exp_name):
    set_seed(config.seed)
    logger = get_logger(log_dir, exp_name)
    logger.info("="*50)
    logger.info("MultiModal Fault Diagnosis Framework")
    logger.info("="*50)
    logger.info(f"使用全部模态: {config.modalities}")
    
    logger.info("Load dataset")
    train_data, val_data, aug_data, test_data = build_dataloader(config, logger)
    logger.info("Training...")
    model = MultiModalDiag(config, logger, log_dir)
    model.train(train_data, val_data, aug_data)
    res: Result = model.evaluate(test_data)
    return res.export_df(exp_name)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='MultiModal Fault Diagnosis Framework')
    parser.add_argument('--dataset', default='gaia', choices=['gaia', 'sn', 'tt'],
                        help='Dataset to use: gaia, sn, or tt')
    parser.add_argument('--gpu', default='0', help='GPU device ID')
    parser.add_argument('--seed', default=None, type=int, help='Random seed (default: use config value)')
    args = parser.parse_args()
    
    dataset = args.dataset
    config = Config(dataset)
    config.gpu_device = args.gpu
    if args.seed is not None:
        config.seed = args.seed
    
    train_and_evaluate(config, f'./logs/{dataset}', f'{dataset}_mmdiag')

