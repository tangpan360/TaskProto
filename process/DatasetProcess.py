import os
import pickle
import torch
import numpy as np
import pandas as pd
import dgl
from core.multimodal_dataset import MultiModalDataSet
from core.aug import aug_drop_node, aug_importance_aware_drop
from config.exp_config import Config
from sklearn.model_selection import train_test_split


class DatasetProcess:
    """Load the preprocessed dataset.pkl and build the graph dataset."""
    
    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        
        # Dataset paths from config.
        self.dataset_path = config.dataset_path
        self.nodes_path = config.nodes_path
        self.edges_path = config.edges_path
        
    def process(self):
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Load data.
        with open(self.dataset_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.logger.info(f"Loaded {len(data_dict)} samples")
        
        # Load real topology (nodes and edges).
        import json
        
        with open(self.nodes_path, 'r') as f:
            nodes_dict = json.load(f)
        with open(self.edges_path, 'r') as f:
            edges_dict = json.load(f)
        
        self.logger.info(f"Loaded real topology: {len(edges_dict['0'])} edges per sample")
        
        # Build label mappings.
        all_services = set()
        all_types = set()
        for sample in data_dict.values():
            all_services.add(sample['fault_service'])
            all_types.add(sample['fault_type'])
        
        all_services = sorted(list(all_services))
        all_types = sorted(list(all_types))
        
        service2idx = {s: i for i, s in enumerate(all_services)}
        type2idx = {t: i for i, t in enumerate(all_types)}
        
        self.logger.info(f"Services: {all_services}")
        self.logger.info(f"Fault types: {all_types}")
        
        # Build datasets.
        train_data = MultiModalDataSet()
        val_data = MultiModalDataSet()
        test_data = MultiModalDataSet()
        
        # Collect all samples and split by data_type.
        train_samples = []
        val_samples = []
        test_samples = []
        
        for sample_id, sample in data_dict.items():
            metric_data = sample['metric_data']
            log_data = sample['log_data']
            trace_data = sample['trace_data']
            
            fault_service = sample['fault_service']
            fault_type = sample['fault_type']
            data_type = sample['data_type']
            
            global_root_id = service2idx[fault_service]
            failure_type_id = type2idx[fault_type]
            
            # Use the sample-specific real topology.
            sample_nodes = nodes_dict[str(sample_id)]
            sample_edges = edges_dict[str(sample_id)]
            
            sample_data = {
                'metric_Xs': metric_data,
                'trace_Xs': trace_data,
                'log_Xs': log_data,
                'global_root_id': global_root_id,
                'failure_type_id': failure_type_id,
                'local_root': fault_service,
                'nodes': sample_nodes,
                'edges': sample_edges
            }
            
            if data_type == 'train':
                train_samples.append(sample_data)
            elif data_type == 'val':
                val_samples.append(sample_data)
            elif data_type == 'test':
                test_samples.append(sample_data)
        
        # For the Gaia dataset, if there are no val samples, split 30% from train as validation.
        if self.config.dataset == 'gaia' and len(val_samples) == 0:
            self.logger.info("Gaia dataset: splitting train samples into train/val")
            train_fault_types = [sample['failure_type_id'] for sample in train_samples]
            
            train_samples, val_samples, _, _ = train_test_split(
                train_samples,
                train_fault_types,
                test_size=0.3,  # 30% for validation
                random_state=self.config.seed,  # fixed random seed for reproducibility
                stratify=train_fault_types  # stratified sampling by fault type
            )
            
            self.logger.info(f"Split with stratification by fault type")
        
        # Build dataset objects.
        for sample_data in train_samples:
            train_data.add_data(**sample_data)
        
        for sample_data in val_samples:
            val_data.add_data(**sample_data)
            
        for sample_data in test_samples:
            test_data.add_data(**sample_data)
        
        # Data augmentation
        AUG_PERCENT = 0.2
        AUG_TIMES = 5

        aug_data = []
        if AUG_TIMES > 0:
            # Importance-aware augmentation (distance-based only).
            use_degree = False
            use_distance = True
            self.logger.info("Data augmentation strategy: importance-aware drop (distance-based only)")
            self.logger.info(f"Generating {AUG_TIMES} augmented samples per training sample")
            
            for time in range(AUG_TIMES):
                for (graph, labels) in train_data:
                    root = graph.ndata['root'].tolist().index(1)
                    
                    aug_graph = aug_importance_aware_drop(
                        graph,
                        root,
                        drop_percent=AUG_PERCENT,
                        use_degree=use_degree,
                        use_distance=use_distance
                    )
                    
                    aug_data.append((aug_graph, labels))
        
        self.logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}, Aug samples: {len(aug_data)}")
        
        return train_data, val_data, aug_data, test_data

