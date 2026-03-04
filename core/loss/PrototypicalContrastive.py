import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalContrastiveLoss(nn.Module):
    """
    Prototype-guided, task-specific contrastive learning.
    Supports independent control for FTI and RCL tasks.
    """
    
    def __init__(self, 
                 num_fti_classes,
                 num_rcl_classes,
                 feature_dim=32,
                 temperature=0.3,
                 initial_momentum=0.5,
                 final_momentum=0.9,
                 warmup_epochs=3,
                 use_fti_contrastive=True,
                 use_rcl_contrastive=True,
                 device='cuda'):
        super().__init__()
        
        self.num_fti_classes = num_fti_classes
        self.num_rcl_classes = num_rcl_classes
        self.temperature = temperature
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.warmup_epochs = warmup_epochs
        self.use_fti_contrastive = use_fti_contrastive
        self.use_rcl_contrastive = use_rcl_contrastive
        self.device = device
        self.current_epoch = 0
        
        # Initialize prototypes based on config (register_buffer is saved/loaded automatically).
        if use_fti_contrastive:
            self.register_buffer(
                'prototypes_fti',
                F.normalize(torch.randn(num_fti_classes, feature_dim), dim=1).to(device)
            )
        else:
            self.prototypes_fti = None
        
        if use_rcl_contrastive:
            self.register_buffer(
                'prototypes_rcl',
                F.normalize(torch.randn(num_rcl_classes, feature_dim), dim=1).to(device)
            )
        else:
            self.prototypes_rcl = None
    
    def set_epoch(self, epoch):
        """Set current epoch (used for adaptive momentum)."""
        self.current_epoch = epoch
    
    def get_current_momentum(self):
        """Adaptive momentum: learn fast early, converge stably later."""
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            return self.initial_momentum + \
                   (self.final_momentum - self.initial_momentum) * progress
        return self.final_momentum
    
    def forward(self, f_fti, e_rcl, type_labels, node_root_labels):
        """
        Args:
            f_fti: [B, D] - graph-level features for the FTI task
            e_rcl: [N, D] - node-level features for the RCL task
            type_labels: [B] - fault type labels
            node_root_labels: [N] - root-cause node labels
        Returns:
            loss_fti, loss_rcl: contrastive losses for the two tasks
        """
        # FTI contrastive loss
        if self.use_fti_contrastive:
            self._update_prototypes(f_fti, type_labels, self.prototypes_fti, self.num_fti_classes)
            loss_fti = self._prototype_loss(f_fti, type_labels, self.prototypes_fti)
        else:
            loss_fti = torch.tensor(0.0, device=self.device)
        
        # RCL contrastive loss
        if self.use_rcl_contrastive:
            self._update_prototypes(e_rcl, node_root_labels, self.prototypes_rcl, self.num_rcl_classes)
            loss_rcl = self._prototype_loss(e_rcl, node_root_labels, self.prototypes_rcl)
        else:
            loss_rcl = torch.tensor(0.0, device=self.device)
        
        return loss_fti, loss_rcl
    
    def _update_prototypes(self, features, labels, prototypes, num_classes):
        """
        Momentum-based prototype update.
        
        Uses an exponential moving average (EMA) to update prototype vectors and
        accumulate global knowledge across batches.
        The momentum coefficient is adapted via get_current_momentum() (optimized for early stopping).
        """
        features = F.normalize(features, dim=1)
        momentum = self.get_current_momentum()
        
        with torch.no_grad():
            for c in range(num_classes):
                mask = (labels == c)
                
                if mask.sum() > 0:
                    # Mean feature for class c in the current batch.
                    class_mean = features[mask].mean(dim=0)
                    class_mean = F.normalize(class_mean.unsqueeze(0), dim=1).squeeze(0)
                    
                    # Standard EMA: prototype_new = m * prototype_old + (1-m) * class_mean
                    prototypes[c] = momentum * prototypes[c] + (1 - momentum) * class_mean
                    
                    # Re-normalize to the unit hypersphere.
                    prototypes[c] = F.normalize(prototypes[c].unsqueeze(0), dim=1).squeeze(0)
    
    def _prototype_loss(self, features, labels, prototypes):
        """Compute sample-to-prototype contrastive loss."""
        # Filter invalid labels (-1 means ignore).
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # Normalize and compute similarity.
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, prototypes.T) / self.temperature
        
        # Cross-entropy loss: pull samples toward their class prototype.
        return F.cross_entropy(logits, labels)
    
    def get_prototype_info(self):
        """Get prototype information (for analysis)."""
        with torch.no_grad():
            info = {}
            
            # FTI prototype info
            if self.use_fti_contrastive and self.prototypes_fti is not None:
                sim_fti = torch.matmul(self.prototypes_fti, self.prototypes_fti.T)
                mask_fti = 1 - torch.eye(self.num_fti_classes, device=self.device)
                avg_sim_fti = (sim_fti * mask_fti).sum() / mask_fti.sum()
                info['fti_inter_similarity'] = avg_sim_fti.item()
                info['prototypes_fti'] = self.prototypes_fti.cpu().numpy()
            
            # RCL prototype info
            if self.use_rcl_contrastive and self.prototypes_rcl is not None:
                sim_rcl = torch.matmul(self.prototypes_rcl, self.prototypes_rcl.T)
                mask_rcl = 1 - torch.eye(self.num_rcl_classes, device=self.device)
                avg_sim_rcl = (sim_rcl * mask_rcl).sum() / mask_rcl.sum()
                info['rcl_inter_similarity'] = avg_sim_rcl.item()
                info['prototypes_rcl'] = self.prototypes_rcl.cpu().numpy()
            
            return info

