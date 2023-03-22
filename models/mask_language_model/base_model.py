from typing import Any, Optional, Union, List
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from utils.metrics import auroc, accuracy, f1
from utils.config_utils import LearningConf, ClassificationParamsConf
from utils.data_utils import masking_all_batches


class BaseModel(pl.LightningModule, ABC):

    def __init__(
        self, output_dim: int,         
        mcc_vocab_size: int,
        amnt_bins: int, 
        *args: Any, 
        **kwargs: Any
        ) -> None:
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.mcc_vocab_size = mcc_vocab_size
        self.amnt_bins = amnt_bins

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
        mcc_seqs, amnt_seqs, _, lengths, avg_amnt, top_mcc = batch


        new_batches, rand_masks = masking_all_batches(
            list_changing_batches=[mcc_seqs, amnt_seqs], 
            mask_tokens = [self.mcc_vocab_size + 1, self.mcc_amnt_seqs + 1],
            masked_rate = 0.15,
            save_tokens = [[0], [0]],
        )

        mcc_seqs, amnt_seqs = new_batches

        logits = self(mcc_seqs, amnt_seqs, lengths, avg_amnt, top_mcc)
        logits = logits.view(-1, self.output_dim)
        labels = mcc_seqs.view(-1)
        preds = torch.argmax(logits, dim=1)

        mask_idx= (rand_masks[0].flatten()).nonzero().view(-1) 

        loss = F.cross_entropy(logits[mask_idx], labels[mask_idx])
        self.log('train_loss', loss)
        
        labels_masked = labels[mask_idx]
        preds_masked = preds[mask_idx]

        return {
            'loss': loss,
            'preds': preds_masked,
            'labels': labels_masked
        }
    

    def training_epoch_end(self, outputs: torch.Tensor) -> Optional[STEP_OUTPUT]:
        preds  = torch.cat([o['preds']  for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])

        self.log('train_accuracy', accuracy(preds, labels), prog_bar=True)
        self.log('train_f1', f1(preds, labels), prog_bar=True)
    

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc = batch

        new_batches, rand_masks = masking_all_batches(
            list_changing_batches=[mcc_seqs, amnt_seqs], 
            mask_tokens = [self.mcc_vocab_size + 1, self.mcc_amnt_seqs + 1],
            masked_rate = 0.15,
            save_tokens = [[0], [0]],
        )

        mcc_seqs, amnt_seqs = new_batches

        logits = self(mcc_seqs, amnt_seqs, lengths, avg_amnt, top_mcc)
        logits = logits.view(-1, self.output_dim)
        labels = mcc_seqs.view(-1)
        mask_idx= (rand_masks[0].flatten()).nonzero().view(-1) 

        loss = F.cross_entropy(logits[mask_idx], labels[mask_idx])
        self.log('val_loss', loss)
        
        labels_masked = labels[mask_idx]
        preds_masked = torch.argmax(logits[mask_idx], dim=1)

        return preds_masked, labels_masked


    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        preds, labels = zip(*outputs)
        preds, labels = torch.cat(preds), torch.cat(labels)
        self.log('val_accuracy', accuracy(preds, labels), prog_bar=True)
        self.log('val_f1', f1(preds, labels), prog_bar=True)


    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc = batch

        new_batches, rand_masks = masking_all_batches(
            list_changing_batches=[mcc_seqs, amnt_seqs], 
            mask_tokens = [self.mcc_vocab_size + 1, self.mcc_amnt_seqs + 1],
            masked_rate = 0.15,
            save_tokens = [[0], [0]],
        )

        mcc_seqs, amnt_seqs = new_batches

        logits = self(mcc_seqs, amnt_seqs, lengths, avg_amnt, top_mcc)
        logits = logits.view(-1, self.output_dim)
        preds = torch.argmax(logits, dim=1)
        labels = mcc_seqs.view(-1)

        mask_idx= (rand_masks[0].flatten()).nonzero().view(-1)   
        labels_masked = labels[mask_idx]
        preds_masked = preds[mask_idx]

        return preds_masked, labels_masked
    

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        preds, labels = zip(*outputs)
        preds, labels = torch.cat(preds), torch.cat(labels)\

        self.log('test_accuracy', accuracy(preds, labels), prog_bar=True)
        self.log('test_f1', f1(preds, labels), prog_bar=True)
