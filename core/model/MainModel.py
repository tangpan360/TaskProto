import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.GraphEncoder import GraphEncoder
from core.model.ModalEncoder import MultiModalEncoder
from core.model.TaskSpecificAttention import AdaptiveModalFusion


class MainModel(nn.Module):
    """
    Main model for multimodal fault diagnosis.
    Pipeline: raw time-series -> multimodal encoder -> graph network -> diagnosis outputs
    """
    def __init__(self, config: Config):
        super(MainModel, self).__init__()
        
        self.config = config
        
        # Multimodal encoder (encodes raw inputs into a fixed dimension).
        self.modal_encoder = MultiModalEncoder(
            output_dim=config.feature_embedding_dim,
            metric_channels=config.metric_channels,
            log_dim=config.log_dim,
            seq_len=config.seq_len
        )
        
        # Graph encoders (one per modality).
        self.graph_encoders = nn.ModuleDict()
        for modality in config.modalities:
            self.graph_encoders[modality] = GraphEncoder(
                feature_embedding_dim=config.feature_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,
                feat_drop=config.feat_drop
            )

        # Simplified modality fusion module.
        self.adaptive_fusion = AdaptiveModalFusion(
            modal_dim=config.graph_out,
            num_heads=getattr(config, 'attention_heads', 4),
            dropout=getattr(config, 'attention_dropout', 0.1),
            fusion_mode=config.fusion_mode
        )
        
        # Unified classifier: after fusion, any modality combination outputs 32-dim features.
        self.typeClassifier = Classifier(
            in_dim=config.graph_out,
            hiddens=config.linear_hidden,
            out_dim=config.ft_num
        )
        self.locator = Voter(
            config.graph_out,
            hiddens=config.linear_hidden,
            out_dim=1
        )

    def forward(self, batch_graphs):
        # Use all modalities.
        used_modalities = self.config.modalities
        
        # Step 1: encode raw inputs with the multimodal encoder.
        metric_raw = batch_graphs.ndata['metric']  # [num_nodes, 20, 12]
        log_raw = batch_graphs.ndata['log']  # [num_nodes, 48]
        trace_raw = batch_graphs.ndata['trace']  # [num_nodes, 20, 2]
        
        metric_emb, log_emb, trace_emb = self.modal_encoder(metric_raw, log_raw, trace_raw)
        
        modal_embs = {
            'metric': metric_emb,
            'log': log_emb,
            'trace': trace_emb
        }
        
        # Step 2: run graph encoders (only for used modalities).
        fs, es = {}, {}
        
        for modality in used_modalities:
            if modality in self.graph_encoders:
                x_d = modal_embs[modality]
                f_d, e_d = self.graph_encoders[modality](batch_graphs, x_d)  # graph-level, node-level
                fs[modality] = f_d
                es[modality] = e_d

        # Step 3: multimodal fusion.
        f, e, fusion_info = self.adaptive_fusion(fs, es, used_modalities)
        # Output: f[B, 32], e[N, 32]

        # Step 4: diagnosis heads.
        type_logit = self.typeClassifier(f)  # fault type identification
        root_logit = self.locator(e)  # root cause localization

        # Store fusion info for analysis.
        self._last_fusion_info = fusion_info
        
        return fs, es, root_logit, type_logit, f, e
    
    def get_fusion_info(self):
        """
        Get fusion information from the last forward pass.
        Used for model analysis and visualization.
        
        Returns:
            dict: fusion weights and attention information
        """
        return getattr(self, '_last_fusion_info', {})
    
    def get_attention_info(self):
        """
        Get attention weight information (backward compatible).
        
        Returns:
            dict: attention weights for FTI and RCL tasks
        """
        fusion_info = self.get_fusion_info()
        attention_info = {}
        
        # Extract attention info from fusion_info.
        if 'fti_attention' in fusion_info:
            attention_info['fti_attention'] = fusion_info['fti_attention']
        if 'rcl_attention' in fusion_info:
            attention_info['rcl_attention'] = fusion_info['rcl_attention']
            
        return attention_info
    
    def get_fusion_mode(self):
        """Get the current fusion mode."""
        return self.adaptive_fusion.fusion_mode
    
    def get_modal_importance_analysis(self, used_modalities):
        """
        Get modality-importance analysis results.
        
        Args:
            used_modalities: list of modality names used in last forward pass
            
        Returns:
            dict: modality-importance analysis for FTI and RCL tasks
        """
        attention_info = getattr(self, '_last_attention_info', {})
        
        if not attention_info or self.adaptive_fusion is None:
            return {'error': 'No attention information available'}
        
        analysis = {}
        
        # Modality importance for the FTI task.
        if 'fti_attention' in attention_info:
            fti_importance = self.adaptive_fusion.get_modal_importance(
                attention_info['fti_attention'], used_modalities
            )
            analysis['fti_modal_importance'] = fti_importance
        
        # Modality importance for the RCL task.
        if 'rcl_attention' in attention_info:
            rcl_importance = self.adaptive_fusion.get_modal_importance(
                attention_info['rcl_attention'], used_modalities
            )
            analysis['rcl_modal_importance'] = rcl_importance
        
        # Compute inter-task modality preference differences.
        if 'fti_modal_importance' in analysis and 'rcl_modal_importance' in analysis:
            fti_imp = analysis['fti_modal_importance']
            rcl_imp = analysis['rcl_modal_importance']
            
            differences = {}
            for modality in used_modalities:
                if modality in fti_imp and modality in rcl_imp:
                    differences[modality] = abs(fti_imp[modality] - rcl_imp[modality])
            
            analysis['task_preference_differences'] = differences
        
        return analysis
