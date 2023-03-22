import torch

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return roc_auc_score(labels.detach().cpu().numpy(), probs.detach().cpu().numpy())

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return accuracy_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())

def f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='weighted')

# TO DO
class TopNAccuracy():
  def __init__(self, N):
    self.N = N
    self.got_right = 0
    self.total = 0

  def update(self, logits, target):
    args_sorted = torch.argsort(logits, descending=True)
    self.total += len(target)
    for i, row in enumerate(args_sorted):
      if target[i] in row[:self.N]:
        self.got_right += 1
  
  def compute(self):
    return self.got_right / self.total
