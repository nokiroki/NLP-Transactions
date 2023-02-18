from typing import Any, Optional

import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from spacecutter.models import LogisticCumulativeLink
from spacecutter.losses import CumulativeLinkLoss


class Transaction2VecJoint(pl.LightningModule):

    def __init__(
        self,
        amnt_loss: str,
        mcc_vocab_size: int,
        mcc_emb_size: int,
        amnt_bins: int,
        amnt_emb_size: int,
        emb_size: int,
        lr: float,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        assert amnt_loss in ('ordinal', 'ce')

        self.mcc_input_embeddings   = nn.Embedding(mcc_vocab_size + 1, mcc_emb_size, padding_idx=0)
        self.amnt_input_embeddings  = nn.Embedding(amnt_bins + 1, amnt_emb_size, padding_idx=0)
        self.hidden_linear          = nn.Linear(mcc_emb_size + amnt_emb_size, emb_size, bias=False)
        self.mcc_output             = nn.Linear(emb_size, mcc_vocab_size, bias=False)
        self.amnt_output            = nn.Linear(
            emb_size,
            1 if amnt_loss == 'ordinal' else amnt_bins,
            bias=True if amnt_loss == 'ordinal' else False
        )
        self.mcc_criterion          = nn.CrossEntropyLoss()

        if amnt_loss == 'ordinal':
            self.amnt_output = nn.Sequential(
                self.amnt_output,
                LogisticCumulativeLink(amnt_bins)
            )
            self.amnt_criterion = CumulativeLinkLoss()
        else:
            self.amnt_criterion = nn.CrossEntropyLoss()

        self.lr = lr
        self.amnt_loss = amnt_loss


    def forward(
        self,
        ctx_mccs: torch.Tensor,
        ctx_amnts: torch.Tensor,
        ctx_lengths: torch.Tensor
    ) -> Any:
        mcc_hidden = self.mcc_input_embeddings(ctx_mccs) / ctx_lengths.view(-1, 1, 1)
        amnt_hidden = self.amnt_input_embeddings(ctx_amnts) / ctx_lengths.view(-1, 1, 1)
        hidden = self.hidden_linear(torch.cat([mcc_hidden, amnt_hidden], -1)).sum(1)
        mcc_logits = self.mcc_output(hidden)
        amnt_logits = self.amnt_output(hidden)
        return mcc_logits, amnt_logits


    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return {'optimizer': optimizer}
    

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        ctx_mccs, ctx_amnts, center_mccs, center_amnts, ctx_lengths = batch
        mcc_logits, amnt_logits = self(ctx_mccs, ctx_amnts, ctx_lengths)
        if self.amnt_loss == 'ordinal':
            center_amnts = center_amnts.view(-1, 1)
        loss = self.mcc_criterion(mcc_logits, center_mccs - 1) + self.amnt_criterion(amnt_logits, center_amnts - 1)
        self.log('train_loss', loss)
        return loss
    

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        ctx_mccs, ctx_amnts, center_mccs, center_amnts, ctx_lengths = batch
        mcc_logits, amnt_logits = self(ctx_mccs, ctx_amnts, ctx_lengths)
        if self.amnt_loss == 'ordinal':
            center_amnts = center_amnts.view(-1, 1)
        loss = self.mcc_criterion(mcc_logits, center_mccs - 1) + self.amnt_criterion(amnt_logits, center_amnts - 1)
        self.log('val_loss', loss, prog_bar=True)
