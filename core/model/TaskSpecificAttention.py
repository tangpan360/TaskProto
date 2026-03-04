import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecificModalAttention(nn.Module):
    """
    Task-specific modality attention mechanism.
    The FTI and RCL tasks use independent learnable query vectors.
    """
    def __init__(self, modal_dim, num_heads=4, dropout=0.1, task_type="fti"):
        super(TaskSpecificModalAttention, self).__init__()
        
        self.task_type = task_type
        self.modal_dim = modal_dim
        self.num_heads = num_heads
        
        # Learnable task query vector (core improvement).
        self.task_query = nn.Parameter(torch.empty(1, modal_dim))
        nn.init.xavier_normal_(self.task_query)
        
        # Simplified query projection.
        self.query_proj = nn.Sequential(
            nn.Linear(modal_dim, modal_dim),
            nn.LayerNorm(modal_dim),
            nn.GELU()
        )
        
        # Key/Value projections.
        self.key_proj = nn.Linear(modal_dim, modal_dim)
        self.value_proj = nn.Linear(modal_dim, modal_dim)
        
        # Multi-head attention.
        self.attention = nn.MultiheadAttention(
            embed_dim=modal_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced output projection.
        self.output_proj = nn.Sequential(
            nn.Linear(modal_dim, modal_dim),
            nn.LayerNorm(modal_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, modal_features, context_features=None):
        """
        Attention computation based on a learnable task query.
        
        Args:
            modal_features: [batch_size, num_modals, modal_dim] stacked modality features
            context_features: kept for backward compatibility (unused)
        
        Returns:
            fused_features: [batch_size, modal_dim] fused features
            attention_weights: [batch_size, num_heads, 1, num_modals] attention weights
        """
        batch_size, num_modals, modal_dim = modal_features.shape
        
        # Use the learnable task query (avoids circular dependency).
        query = self.task_query.expand(batch_size, -1)  # [batch_size, modal_dim]
        query = self.query_proj(query).unsqueeze(1)  # [batch_size, 1, modal_dim]
        
        # Key/Value transform.
        key_features = self.key_proj(modal_features)    # [batch_size, num_modals, modal_dim]
        value_features = self.value_proj(modal_features)  # [batch_size, num_modals, modal_dim]
        
        # Multi-head attention.
        attn_output, attn_weights = self.attention(
            query=query,                    # [batch_size, 1, modal_dim]
            key=key_features,              # [batch_size, num_modals, modal_dim]
            value=value_features,          # [batch_size, num_modals, modal_dim]
            need_weights=True
        )
        
        # Output projection.
        fused_features = self.output_proj(attn_output.squeeze(1))  # [batch_size, modal_dim]
        
        return fused_features, attn_weights


class AdaptiveModalFusion(nn.Module):
    """
    Simplified multimodal fusion module: two fusion strategies with a unified 32-dim output.
    """
    def __init__(self, modal_dim, num_heads=4, dropout=0.1, fusion_mode="adaptive"):
        super(AdaptiveModalFusion, self).__init__()
        
        self.fusion_mode = fusion_mode
        self.modal_dim = modal_dim
        
        if fusion_mode == "average":
            # Simple average fusion.
            pass
            
        elif fusion_mode == "adaptive":
            # Adaptive weighted fusion with task-specific attention.
            self.fti_attention = TaskSpecificModalAttention(
                modal_dim=modal_dim,
                num_heads=num_heads,
                dropout=dropout,
                task_type="fti"
            )
            
            self.rcl_attention = TaskSpecificModalAttention(
                modal_dim=modal_dim,
                num_heads=num_heads,
                dropout=dropout,
                task_type="rcl"
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}. Use 'average' or 'adaptive'.")
    
    def forward(self, modal_fs, modal_es, used_modalities):
        """
        Multimodal fusion for fair comparison.
        
        Args:
            modal_fs: dict of {modality: [batch_size, modal_dim]} graph-level features
            modal_es: dict of {modality: [num_nodes, modal_dim]} node-level features
            used_modalities: list of modality names
        
        Returns:
            f_fused: [batch_size, modal_dim] features for the FTI task (unified 32-dim output)
            e_fused: [num_nodes, modal_dim] features for the RCL task (unified 32-dim output)
            fusion_info: dict fusion process information
        """
        fusion_info = {}
        
        # Step 1: stack modality features into a unified format.
        f_stack = torch.stack([modal_fs[mod] for mod in used_modalities], dim=1)  # [B, M, D]
        e_stack = torch.stack([modal_es[mod] for mod in used_modalities], dim=1)  # [N, M, D]
        
        if self.fusion_mode == "average":
            # Simple average fusion.
            f_fused = f_stack.mean(dim=1)  # [B, D]
            e_fused = e_stack.mean(dim=1)  # [N, D]
            fusion_info['fusion_type'] = 'simple_average'
            
        elif self.fusion_mode == "adaptive":
            # Adaptive weighted fusion (using learnable task queries).
            f_fused, fti_attn = self.fti_attention(f_stack)  # [B, D]
            e_fused, rcl_attn = self.rcl_attention(e_stack)  # [N, D]
            
            fusion_info['fti_attention'] = fti_attn
            fusion_info['rcl_attention'] = rcl_attn  
            fusion_info['fusion_type'] = 'adaptive_attention'
            
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}. Use 'average' or 'adaptive'.")
        
        # Both modes output the same dimensions: [B, D] and [N, D].
        return f_fused, e_fused, fusion_info
    
    def get_modal_importance(self, attention_weights, modalities):
        """
        Extract modality-importance scores from attention weights.
        
        Args:
            attention_weights: [batch_size, num_heads, 1, num_modals] 
            modalities: list of modality names
            
        Returns:
            dict: {modality: importance_score}
        """
        if attention_weights is None:
            return {}
        
        # Compute mean attention weights.
        avg_weights = attention_weights.mean(dim=1).squeeze(1)  # [batch_size, num_modals]
        avg_weights = avg_weights.mean(dim=0)  # [num_modals] - average across batch
        
        # Create modality-importance dict.
        importance = {}
        for i, modality in enumerate(modalities):
            if i < len(avg_weights):
                importance[modality] = avg_weights[i].item()
        
        return importance


class ModalAttentionVisualizer:
    """
    Attention weight visualization utility.
    """
    @staticmethod
    def plot_attention_comparison(fti_importance, rcl_importance, modalities, save_path=None):
        """
        Compare modality attention weights between FTI and RCL tasks.
        
        Args:
            fti_importance: dict of {modality: weight}
            rcl_importance: dict of {modality: weight}  
            modalities: list of modality names
            save_path: str, optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            x = np.arange(len(modalities))
            width = 0.35
            
            fti_weights = [fti_importance.get(mod, 0) for mod in modalities]
            rcl_weights = [rcl_importance.get(mod, 0) for mod in modalities]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            rects1 = ax.bar(x - width/2, fti_weights, width, label='FTI Task', alpha=0.8)
            rects2 = ax.bar(x + width/2, rcl_weights, width, label='RCL Task', alpha=0.8)
            
            ax.set_xlabel('Modalities')
            ax.set_ylabel('Attention Weight')
            ax.set_title('Modal Attention Weights Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([mod.capitalize() for mod in modalities])
            ax.legend()
            
            # Add numeric labels.
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Please install it for visualization.")
        except Exception as e:
            print(f"Visualization error: {e}")