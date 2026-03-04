import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score



def RCA_eval(root_logit, num_nodes_list, roots):
    res = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [], "HR@5": [], "MRR@3": []}
    
    start_idx = 0
    for idx, num_nodes in enumerate(num_nodes_list):
        end_idx = start_idx + num_nodes
        node_logits = root_logit[start_idx : end_idx].reshape(1, -1)
        root = roots[start_idx : end_idx].tolist().index(1)

        _, sorted_indices = torch.sort(node_logits, descending=True)
        for j in range(1, 6):
            # HR@k
            if root in sorted_indices.flatten()[:j]:
                res[f"HR@{j}"].append(1)
            else:
                res[f"HR@{j}"].append(0)
        # MRR
        rank = (sorted_indices == root).nonzero(as_tuple=True)[1].item() + 1
        if rank <= 3:
            res["MRR@3"].append(1 / rank)
        else:
            res["MRR@3"].append(0)

        start_idx += num_nodes
    for k in range(1, 6):
        res[f'HR@{k}'] = np.sum(res[f'HR@{k}'])/len(num_nodes_list)
    res['MRR@3'] = np.sum(res['MRR@3'])/len(num_nodes_list)
    return res
    
        
def FTI_eval(output, target, k=5):
    k = min(k, output.size(1))
    res = {"pre": [], "rec": [], "f1": []}
    # NOTE:
    # - pre/rec/f1 here are for Failure Type Identification (FTI), i.e., single-label multi-class classification.
    # - We report macro-averaged precision/recall/F1 (each class equally weighted).
    # - IMPORTANT: macro-F1 is NOT generally equal to 2*macro-P*macro-R/(macro-P+macro-R),
    #   so we compute it directly via sklearn.metrics.f1_score with average='macro'.
    res['pre'] = precision(output, target, k)
    res['rec'] = recall(output, target, k)
    res['f1'] = f1score(output, target, k)
    return res



def target_rank(output, target, k=10):
    k = min(k, output.size(1))
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    ranks = []
    for col in range(correct.size(1)):
        try:
            idx=torch.where(correct[:, col] == target[col])[0].item() + 1
        except:
            idx=k
        ranks.append(idx)
    
    return ranks


def precision(output, target, k=5):
    k = min(k, output.size(1))
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    pre = precision_score(y_true, y_pred[:, 0], average='macro', zero_division=0)

    return pre


def recall(output, target, k=5):
    k = min(k, output.size(1))
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    rec = recall_score(y_true, y_pred[:, 0], average='macro', zero_division=0)

    return rec


def f1score(output, target, k=5):
    k = min(k, output.size(1))
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    f1 = f1_score(y_true, y_pred[:, 0], average='macro', zero_division=0)

    return f1