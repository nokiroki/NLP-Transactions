import torch

from sklearn.metrics import roc_auc_score

def auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return roc_auc_score(labels.detach().cpu().numpy(), probs.detach().cpu().numpy())
