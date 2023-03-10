from typing import Any, Optional, Union, List
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from utils.metrics import auroc
from utils.config_utils import LearningConf, ClassificationParamsConf


class BaseModel(pl.LightningModule, ABC):

    @abstractmethod
    def forward(
        self,
        mcc_seqs: torch.Tensor,
        amnt_weights: torch.Tensor,
        lengths: torch.Tensor,
        avg_amnt: Optional[torch.Tensor],
        top_mcc: Optional[torch.Tensor],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        raise NotImplementedError()
    

    def _mean_pooling(self, outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_length = outputs.size(1)
        mask = torch.vstack(
            [torch.cat([torch.zeros(length), 
            torch.ones(max_length - length)]) for length in lengths]
        )
        mask = mask.bool().to(outputs.device).unsqueeze(-1)
        outputs.masked_fill_(mask, 0)
        feature_vector = outputs.sum(1) / lengths.unsqueeze(-1)
        return feature_vector
    

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc = batch
        logits = self(mcc_seqs, amnt_seqs, lengths, avg_amnt, top_mcc)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log('train_loss', loss)
        return {'loss': loss, 'probs': torch.sigmoid(logits), 'labels': labels}
    

    def training_epoch_end(self, outputs: torch.Tensor) -> Optional[STEP_OUTPUT]:
        probs  = torch.cat([o['probs']  for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])
        self.log('train_auroc', auroc(probs, labels), prog_bar=True)
    

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc = batch
        logits = self(mcc_seqs, amnt_seqs, lengths, avg_amnt, top_mcc)
        probs = torch.sigmoid(logits)
        return probs, labels


    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        probs, labels = zip(*outputs)
        probs, labels = torch.cat(probs), torch.cat(labels)
        self.log('val_auroc', auroc(probs, labels), prog_bar=True)


    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc = batch
        logits = self(mcc_seqs, amnt_seqs, lengths, avg_amnt, top_mcc)
        probs = torch.sigmoid(logits)
        return probs, labels
    

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        probs, labels = zip(*outputs)
        probs, labels = torch.cat(probs), torch.cat(labels)
        self.log('test_auroc', auroc(probs, labels))
