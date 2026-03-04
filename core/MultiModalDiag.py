import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import dgl
import numpy as np

from core.loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from core.loss.PrototypicalContrastive import PrototypicalContrastiveLoss
from core.model.MainModel import MainModel
from utils.eval import *
from utils.early_stop import EarlyStopping
from utils.Result import Result
from config.exp_config import Config


class MultiModalDiag(object):
    """Training and evaluation framework for multimodal fault diagnosis."""

    def __init__(self, config: Config, logger, log_dir: str):
        self.config = config
        self.logger = logger
        os.makedirs(log_dir, exist_ok=True)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Currently using GPU {}".format(config.gpu_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device
            self.device = 'cuda'
        else:
            logger.info("Currently using CPU (GPU is highly recommended)")
            self.device = 'cpu'

        self.result = Result()
        self.writer = SummaryWriter(log_dir)
        self.printParams()
        
        # Prototype-guided contrastive learning.
        # FTI contrastive: ON; RCL contrastive: OFF.
        self.contrast_loss = PrototypicalContrastiveLoss(
            num_fti_classes=config.n_type,
            num_rcl_classes=config.n_instance,
            feature_dim=config.graph_out,
            temperature=config.temperature,
            initial_momentum=config.initial_momentum,
            final_momentum=config.final_momentum,
            warmup_epochs=config.warmup_epochs,
            use_fti_contrastive=True,
            use_rcl_contrastive=False,
            device=self.device
        )

    def printParams(self):
        self.config.print_configs(self.logger)

    def train(self, train_data, val_data, aug_data):
        AUG_TIMES = 5

        model = MainModel(self.config).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
        # Automatic task-weighting: loss = AWL(l_rcl, l_fti, l_fti_contrast)
        awl = AutomaticWeightedLoss(3)

        self.logger.info(model)
        self.logger.info(f"Start training for {self.config.epochs} epochs.")
        
        train_times = []
        Z_r2fs, Z_f2rs = [], []
        
        earlyStop = EarlyStopping(patience=self.config.patience)
        best_model_path = os.path.join(self.writer.log_dir, 'MMDiag_best.pt')
        
        for epoch in range(self.config.epochs):
            # Set current epoch (for adaptive momentum in prototype contrastive learning).
            self.contrast_loss.set_epoch(epoch)
            
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss, epoch_rcl_l, epoch_fti_l = 0, 0, 0
            epoch_rcl_contrast, epoch_fti_contrast = 0, 0  # contrastive loss tracking
            rcl_results = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [],"HR@5": [], "MRR@3": []}
            fti_results = {'pre':[], 'rec':[], 'f1':[]}

            train_dl = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate)
            for batch_graphs, batch_labels in train_dl:
                batch_graphs = batch_graphs
                instance_labels = batch_labels[:, 0]
                type_labels = batch_labels[:, 1]

                if AUG_TIMES > 0:
                    raw_graphs = dgl.unbatch(batch_graphs)
                    aug_graphs, aug_labels = map(list, zip(*random.sample(aug_data, len(raw_graphs))))
                    batch_graphs = dgl.batch(raw_graphs + aug_graphs)
                    instance_labels = torch.hstack((instance_labels, torch.tensor(aug_labels)[:,0].flatten()))
                    type_labels = torch.hstack((type_labels, torch.tensor(aug_labels)[:,1].flatten()))

                batch_graphs = batch_graphs.to(self.device)
                instance_labels = instance_labels.to(self.device)
                type_labels = type_labels.to(self.device)

                opt.zero_grad()
                fs, es, root_logit, type_logit, f_fused, e_fused = model(batch_graphs)

                # Primary task losses.
                l_rcl = self.cal_rcl_loss(root_logit, batch_graphs)
                l_fti = F.cross_entropy(type_logit, type_labels)
                
                # Task-specific contrastive loss (FTI only)
                node_root_labels = self._expand_labels_to_nodes(instance_labels, batch_graphs)
                l_fti_contrast, _ = self.contrast_loss(
                    f_fused,           # FTI graph-level features [B, D]
                    e_fused,           # RCL node-level features [N, D]
                    type_labels,       # [B]
                    node_root_labels   # [N]
                )

                # Total loss
                total_loss = awl(l_rcl, l_fti, l_fti_contrast)

                total_loss.backward()
                opt.step()
                
                epoch_loss += total_loss.detach().item()
                epoch_rcl_l += l_rcl.detach().item()
                epoch_fti_l += l_fti.detach().item()
                epoch_fti_contrast += l_fti_contrast.detach().item()

                rcl_res = RCA_eval(root_logit, batch_graphs.batch_num_nodes(), batch_graphs.ndata['root'])
                fti_res = FTI_eval(type_logit, type_labels)
                [rcl_results[key].append(value) for key, value in rcl_res.items()]
                [fti_results[key].append(value) for key, value in fti_res.items()]
                n_iter += 1
                
            mean_epoch_loss = epoch_loss / n_iter
            mean_rcl_loss = epoch_rcl_l / n_iter
            mean_fti_loss = epoch_fti_l / n_iter
            end_time = time.time()
            time_per_epoch = (end_time - start_time)
            train_times.append(time_per_epoch)
            
            self.logger.info(
                "Epoch {} done. Loss: {:.3f}, RCL: {:.3f}, FTI: {:.3f}, FTI_Contrast: {:.3f}, Time: {:.3f}[s]".format(
                    epoch, mean_epoch_loss, mean_rcl_loss, mean_fti_loss, epoch_fti_contrast / n_iter, time_per_epoch
                )
            )

            for k, v in rcl_results.items():
                rcl_results[k] = np.mean(v)
            for k, v in fti_results.items():
                fti_results[k] = np.mean(v)
                
            self.writer.add_scalar('loss/train_total_loss', mean_epoch_loss, global_step=epoch)
            self.writer.add_scalar('train/HR@3', rcl_results['HR@3'], global_step=epoch)
            self.writer.add_scalar('train/f1-score', fti_results['f1'], global_step=epoch)
            
            # Evaluate on the validation set.
            val_loss, val_rcl, val_fti = self._validate(model, val_data)
            
            self.writer.add_scalar('loss/val_total_loss', val_loss, global_step=epoch)
            self.writer.add_scalar('val/HR@3', val_rcl['HR@3'], global_step=epoch)
            self.writer.add_scalar('val/f1-score', val_fti['f1'], global_step=epoch)
            
            self.logger.info(f"Val Loss: {val_loss:.3f}, Val HR@3: {val_rcl['HR@3']:.3%}, Val F1: {val_fti['f1']:.3%}")

            # Early stopping (based on validation loss).
            stop, is_best = earlyStop.should_stop(val_loss, epoch)
            
            if is_best:
                # Save best model.
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'val_loss': val_loss,
                }
                torch.save(state, best_model_path)
                self.logger.info(f"✓ Best model saved at epoch {epoch} with val_loss: {val_loss:.3f}")
            
            if stop:
                self.logger.info(f"Early stop at epoch {epoch} due to no improvement on validation set.")
                break

        # Save final model (last epoch).
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }
        torch.save(state, os.path.join(self.writer.log_dir, 'MMDiag_last.pt'))
        self.result.set_train_efficiency(train_times)
        self.logger.info("Training has finished.")
        self.logger.info(f"Best model saved at: {best_model_path}")

    def _validate(self, model, val_data):
        """
        Evaluate the model on the validation set.
        
        Returns:
            tuple: (val_loss, rcl_results, fti_results)
        """
        model.eval()
        val_loss, val_rcl_l, val_fti_l = 0, 0, 0
        rcl_results = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [], "HR@5": [], "MRR@3": []}
        fti_results = {'pre': [], 'rec': [], 'f1': []}
        n_iter = 0
        
        val_dl = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False, collate_fn=self.collate)
        
        with torch.no_grad():
            for batch_graphs, batch_labels in val_dl:
                batch_graphs = batch_graphs.to(self.device)
                instance_labels = batch_labels[:, 0].to(self.device)
                type_labels = batch_labels[:, 1].to(self.device)
                
                fs, es, root_logit, type_logit, _, _ = model(batch_graphs)
                
                # Only compute primary task losses.
                l_rcl = self.cal_rcl_loss(root_logit, batch_graphs)
                l_fti = F.cross_entropy(type_logit, type_labels)
                
                total_loss = l_rcl + l_fti
                
                val_loss += total_loss.detach().item()
                val_rcl_l += l_rcl.detach().item()
                val_fti_l += l_fti.detach().item()
                
                rcl_res = RCA_eval(root_logit, batch_graphs.batch_num_nodes(), batch_graphs.ndata['root'])
                fti_res = FTI_eval(type_logit, type_labels)
                [rcl_results[key].append(value) for key, value in rcl_res.items()]
                [fti_results[key].append(value) for key, value in fti_res.items()]
                n_iter += 1
        
        mean_val_loss = val_loss / n_iter
        for k, v in rcl_results.items():
            rcl_results[k] = np.mean(v)
        for k, v in fti_results.items():
            fti_results[k] = np.mean(v)
        
        model.train()  # restore train mode
        return mean_val_loss, rcl_results, fti_results

    def evaluate(self, test_data, model=None):
        if model is None:
            # Load best checkpoint if present, otherwise fallback to last.
            model_path = os.path.join(self.writer.log_dir, 'MMDiag_best.pt')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.writer.log_dir, 'MMDiag_last.pt')
                self.logger.info("Best model not found, fallback to last model")
            
            self.logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path)
            
            model = MainModel(self.config).to(self.device)
            model.load_state_dict(checkpoint['model'])
            self.logger.info(f"Model loaded from epoch {checkpoint['epoch']}")
            
        model.eval()
        root_logits, type_logits = [], []
        roots, types = [], []
        inference_times = []
        num_node_list = []
        
        for data in test_data:
            graph = data[0].to(self.device)
            failure_type = data[1][1]
            roots.append(graph.ndata['root'])
            types.append(failure_type)
            num_node_list.append(graph.num_nodes())
        
            start_time = time.time()
            with torch.no_grad():
                _, _, root_logit, type_logit, _, _ = model(graph)
                root_logits.append(root_logit.flatten())
                type_logits.append(type_logit.flatten())
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
        root_logits = torch.hstack(root_logits).cpu()
        type_logits = torch.vstack(type_logits).cpu()
        roots = torch.hstack(roots)
        types = torch.tensor(types)

        rcl_res = RCA_eval(root_logits, num_node_list, roots)
        fti_res = FTI_eval(type_logits, types)
        self.result.set_performance(rcl_res, fti_res)
        self.result.set_inference_efficiency(inference_times)

        avg_3 = np.mean([rcl_res['HR@1'], rcl_res['HR@2'], rcl_res['HR@3']])

        self.logger.info("[Root localization] HR@1: {:.3%}, HR@2: {:.3%}, HR@3: {:.3%}, HR@4: {:.3%}, HR@5: {:.3%}, avg@3: {:.3f}, MRR@3: {:.3f}"\
            .format(rcl_res['HR@1'], rcl_res['HR@2'], rcl_res['HR@3'], rcl_res['HR@4'], rcl_res['HR@5'] , avg_3, rcl_res['MRR@3']))
        self.logger.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}"\
            .format(fti_res['pre'], fti_res['rec'], fti_res['f1']))
        self.logger.info(f"The average test time is {np.mean(inference_times)}[s]")

        return self.result

    def cal_rcl_loss(self, root_logit, batch_graphs):        
        num_nodes_list = batch_graphs.batch_num_nodes()
        total_loss = None
        
        start_idx = 0
        for idx, num_nodes in enumerate(num_nodes_list):
            end_idx = start_idx + num_nodes
            node_logits = root_logit[start_idx : end_idx].reshape(1, -1)
            root = batch_graphs.ndata["root"][start_idx : end_idx].tolist().index(1)
            loss = F.cross_entropy(node_logits, torch.LongTensor([root]).view(1).to(self.device))
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            start_idx += num_nodes
            
        l_rcl = total_loss / len(num_nodes_list)
        return l_rcl

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels
    
    def _expand_labels_to_nodes(self, instance_labels, batch_graphs):
        """
        Expand graph-level labels to node-level labels (only for root-cause nodes).
        
        Args:
            instance_labels: [batch_size] - root-cause label (service ID) for each graph
            batch_graphs: DGL batch graph
        
        Returns:
            node_labels: [num_nodes_total] - root-cause labels per node
                         root-cause node: instance_label (0~n_instance-1)
                         non-root node: -1 (ignored in loss computation)
        """
        node_labels = []
        start_idx = 0
        
        for i, num_nodes in enumerate(batch_graphs.batch_num_nodes()):
            end_idx = start_idx + num_nodes
            
            # Root mask for the current graph [num_nodes].
            root_mask = batch_graphs.ndata['root'][start_idx:end_idx]
            
            # Initialize all node labels to -1 (ignore).
            graph_labels = [-1] * num_nodes
            
            # Find the root-cause node (where root == 1).
            root_indices = (root_mask == 1).nonzero(as_tuple=True)[0]
            
            if len(root_indices) > 0:
                # Assign the real instance_label only to the root-cause node.
                root_idx = root_indices[0].item()  # typically only one root per graph
                graph_labels[root_idx] = instance_labels[i].item()
            
            node_labels.extend(graph_labels)
            start_idx = end_idx
        
        return torch.tensor(node_labels, dtype=torch.long, device=self.device)
