from typing import Any, Optional, Union, List
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from utils.metrics import auroc, accuracy, f1


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
        loss = F.cross_entropy(logits, labels)

        self.log('train_loss', loss)
        return {'loss': loss, 'preds': torch.argmax(logits, dim=1), 'labels': labels}
    

    def training_epoch_end(self, outputs: torch.Tensor) -> Optional[STEP_OUTPUT]:
        preds  = torch.cat([o['preds']  for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])

        self.log('train_accuracy', accuracy(preds, labels), prog_bar=True)
        self.log('train_f1', f1(preds, labels), prog_bar=True)
    

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc = batch
        logits = self(mcc_seqs, amnt_seqs, lengths, avg_amnt, top_mcc)

        preds = torch.argmax(logits, dim=1)
        return preds, labels


    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        preds, labels = zip(*outputs)
        preds, labels = torch.cat(preds), torch.cat(labels)
        self.log('val_accuracy', accuracy(preds, labels), prog_bar=True)
        self.log('val_f1', f1(preds, labels), prog_bar=True)


    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc = batch
        logits = self(mcc_seqs, amnt_seqs, lengths, avg_amnt, top_mcc)
        preds = torch.argmax(logits, dim=1)
        return preds, labels
    

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        preds, labels = zip(*outputs)
        preds, labels = torch.cat(preds), torch.cat(labels)
        self.log('test_accuracy', accuracy(preds, labels))
        self.log('test_f1', f1(preds, labels))

