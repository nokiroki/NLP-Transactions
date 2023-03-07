import torch

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return roc_auc_score(labels.detach().cpu().numpy(), probs.detach().cpu().numpy())

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())

def f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')
