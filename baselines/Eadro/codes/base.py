import os
import time
import copy
import numpy as np

import torch
from torch import nn
import logging

from model import MainModel
from sklearn.metrics import ndcg_score

class BaseModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, lr=1e-3, epoches=50, patience=5, result_dir='./', hash_id=None, enable_fault_classification=False, **kwargs):
        super(BaseModel, self).__init__()
        
        self.epoches = epoches
        self.lr = lr
        self.patience = patience # > 0: use early stop
        self.device = device
        self.enable_fault_classification = enable_fault_classification

        self.model_save_dir = os.path.join(result_dir, hash_id)
        self.model = MainModel(event_num, metric_num, node_num, device, enable_fault_classification=enable_fault_classification, **kwargs)
        self.model.to(device)
    
    def evaluate(self, test_loader, datatype="Test"):
        self.model.eval()
        hrs = np.zeros(5)
        mrrs = []  # MRR@3
        TP, TN, FP, FN = 0, 0, 0, 0
        batch_cnt, epoch_loss = 0, 0.0 
        detect_loss_sum, locate_loss_sum = 0.0, 0.0
        
        # 故障分类评估
        fault_type_preds = []
        fault_type_truths = []
        
        with torch.no_grad():
            for graph, ground_truths, fault_types in test_loader:
                # 根据是否启用故障分类来决定是否传递fault_types
                if self.enable_fault_classification:
                    res = self.model.forward(graph.to(self.device), ground_truths, fault_types)
                    # 收集故障分类预测和真实值
                    if "fault_type_pred" in res:
                        fault_type_preds.extend(res["fault_type_pred"])
                        fault_type_truths.extend(fault_types.cpu().numpy())
                else:
                    res = self.model.forward(graph.to(self.device), ground_truths, None)
                for idx, faulty_nodes in enumerate(res["y_pred"]):
                    culprit = ground_truths[idx].item()
                    if culprit == -1:
                        if faulty_nodes[0] == -1: TN+=1
                        else: FP += 1
                    else:
                        if faulty_nodes[0] == -1: FN+=1
                        else: 
                            TP+=1
                            rank = list(faulty_nodes).index(culprit)
                            # HR@1/2/3/4/5
                            for j in range(5):
                                hrs[j] += int(rank <= j)
                            # MRR@3 (与TVDiag一致)
                            actual_rank = rank + 1  # rank从0开始，实际排名从1开始
                            if actual_rank <= 3:
                                mrrs.append(1.0 / actual_rank)
                            else:
                                mrrs.append(0)
                epoch_loss += res["loss"].item()
                detect_loss_sum += res["detect_loss"].item()
                locate_loss_sum += res["locate_loss"].item()
                batch_cnt += 1
        
        pos = TP+FN
        avg_loss = epoch_loss / batch_cnt if batch_cnt > 0 else 0
        avg_detect_loss = detect_loss_sum / batch_cnt if batch_cnt > 0 else 0
        avg_locate_loss = locate_loss_sum / batch_cnt if batch_cnt > 0 else 0
        
        eval_results = {
                "loss": avg_loss,
                "detect_loss": avg_detect_loss,
                "locate_loss": avg_locate_loss}
        
        # 添加 HR@1/2/3/4/5
        for j in range(1, 6):
            eval_results["HR@"+str(j)] = hrs[j-1]*1.0/pos if pos > 0 else 0
        
        # 添加 MRR@3
        eval_results["MRR@3"] = np.mean(mrrs) if len(mrrs) > 0 else 0
        
        # 添加 avg@3
        eval_results["avg@3"] = np.mean([eval_results["HR@1"], eval_results["HR@2"], eval_results["HR@3"]])
        
        # 添加故障分类评估（只在启用时计算）
        if self.enable_fault_classification and len(fault_type_preds) > 0 and len(fault_type_truths) > 0:
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            
            # 计算故障分类指标
            fault_acc = accuracy_score(fault_type_truths, fault_type_preds)
            fault_precision = precision_score(fault_type_truths, fault_type_preds, average='macro', zero_division=0)
            fault_recall = recall_score(fault_type_truths, fault_type_preds, average='macro', zero_division=0)
            fault_f1 = f1_score(fault_type_truths, fault_type_preds, average='macro', zero_division=0)
            
            eval_results["Fault_Acc"] = fault_acc
            eval_results["Fault_Pre"] = fault_precision
            eval_results["Fault_Rec"] = fault_recall  
            eval_results["Fault_F1"] = fault_f1
            
        logging.info("{} -- {}".format(datatype, ", ".join([k+": "+str(f"{v:.4f}") for k, v in eval_results.items()])))

        return eval_results
    
    def fit(self, train_loader, val_loader, test_loader, evaluation_epoch=10):
        """使用验证集locate_loss进行模型选择和早停"""
        best_val_locate_loss = float("inf")
        coverage, best_state, eval_res = None, None, None
        worse_count = 0  # 验证集locate_loss不下降计数

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)
        
        for epoch in range(1, self.epoches+1):
            # 训练
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            for graph, label, fault_type in train_loader:
                optimizer.zero_grad()
                # 根据是否启用故障分类来决定是否传递fault_type
                if self.enable_fault_classification:
                    loss = self.model.forward(graph.to(self.device), label, fault_type)['loss']
                else:
                    loss = self.model.forward(graph.to(self.device), label, None)['loss']
                loss.backward()
                # if self.debug:
                #     for name, parms in self.model.named_parameters():
                #         if name=='encoder.graph_model.net.weight':
                #             print(name, "--> grad:",parms.grad)
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start

            epoch_loss = epoch_loss / batch_cnt
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches, epoch_loss, epoch_time_elapsed))

            # 每个epoch在验证集上评估
            val_results = self.evaluate(val_loader, datatype="Val")
            val_locate_loss = val_results["locate_loss"]
            
            # 基于验证集locate_loss进行模型选择和早停
            if val_locate_loss < best_val_locate_loss:
                best_val_locate_loss = val_locate_loss
                eval_res, coverage = val_results, epoch
                best_state = copy.deepcopy(self.model.state_dict())
                self.save_model(best_state)
                worse_count = 0
                logging.info("  → New best Val locate_loss: {:.5f}".format(best_val_locate_loss))
            else:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch {} (val locate_loss not improved for {} epochs)".format(epoch, self.patience))
                    break
        
        # 训练结束，加载最佳模型在测试集上最终评估
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logging.info("\n* Best model from epoch {} with Val locate_loss: {:.5f}".format(coverage, best_val_locate_loss))
            
            # 在测试集上最终评估
            test_results = self.evaluate(test_loader, datatype="Test")
            logging.info("* Test results: " + 
                        ", ".join(["{}: {:.4f}".format(k, v) for k, v in test_results.items()]))
            eval_res = test_results
        else:
            logging.info("Unable to convergence!")

        return eval_res, coverage
    
    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def save_model(self, state, file=None):
        if file is None: file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, file)
