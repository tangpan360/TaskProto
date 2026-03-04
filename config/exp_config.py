from utils.template_utils import get_log_template_count

class Config:
    def __init__(self, dataset) -> None:
        # base config
        self.dataset = dataset
        self.gpu_device = '0'
        self.seed = 12

        self.modalities = ['metric', 'trace', 'log']

        # Task-specific contrastive learning config
        self.temperature = 0.3                # contrastive temperature
        
        # Prototype contrastive learning config
        self.initial_momentum = 0.5          # initial momentum (learn fast early)
        self.final_momentum = 0.9            # final momentum (stable convergence)
        self.warmup_epochs = 3               # momentum warmup epochs

        # model config
        self.batch_size = 8
        self.epochs = 500
        self.feature_embedding_dim = 128  # Eadro encoder output feature dimension
        self.graph_hidden_dim = 64
        self.graph_out = 32
        self.graph_layers = 2
        self.linear_hidden = [64]
        self.lr = 0.001
        self.weight_decay = 0.0001
        
        # Modality fusion config
        self.fusion_mode = "adaptive"       # fusion mode: "average" | "adaptive"
        self.attention_heads = 4            # number of attention heads
        self.attention_dropout = 0.1        # attention dropout rate        

        if self.dataset == 'gaia':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 5
            self.aggregator = 'mean'
            # Gaia dataset dimensions
            self.metric_channels = 12
            self.log_dim = get_log_template_count('gaia')
            self.seq_len = 20
            # Gaia dataset paths
            self.dataset_path = "./data/processed_data/gaia/dataset.pkl"
            self.nodes_path = "./data/processed_data/gaia/graph/nodes_static_no_influence.json"
            self.edges_path = "./data/processed_data/gaia/graph/edges_static_no_influence.json"
            # Class counts
            self.n_type = 5          # number of fault types
            self.n_instance = 10     # number of services (root-cause classes)
        elif self.dataset == 'sn':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 3  # number of fault types (Fault Type Number)
            self.aggregator = 'mean'
            self.batch_size = 8
            # SN dataset dimensions
            self.metric_channels = 7
            self.log_dim = get_log_template_count('sn')
            self.seq_len = 10
            # SN dataset paths
            self.dataset_path = "./data/processed_data/sn/dataset.pkl"
            self.nodes_path = "./data/processed_data/sn/graph/nodes_predefined_static_no_influence.json"
            self.edges_path = "./data/processed_data/sn/graph/edges_predefined_static_no_influence.json"
            # Class counts
            self.n_type = 3          # number of fault types
            self.n_instance = 12     # number of services (root-cause classes)
        elif self.dataset == 'tt':
            self.feat_drop = 0
            self.patience = 10
            self.ft_num = 3  # number of fault types (Fault Type Number)
            self.aggregator = 'mean'
            self.batch_size = 8
            # TT dataset dimensions
            self.metric_channels = 7
            self.log_dim = get_log_template_count('tt')
            self.seq_len = 10
            # TT dataset paths
            self.dataset_path = "./data/processed_data/tt/dataset.pkl"
            self.nodes_path = "./data/processed_data/tt/graph/nodes_predefined_static_no_influence.json"
            self.edges_path = "./data/processed_data/tt/graph/edges_predefined_static_no_influence.json"
            # Class counts
            self.n_type = 3          # number of fault types
            self.n_instance = 27     # number of services (root-cause classes)
        else:
            raise NotImplementedError()
    
    def print_configs(self, logger):
        for attr, value in vars(self).items():
            logger.info(f"{attr}: {value}")