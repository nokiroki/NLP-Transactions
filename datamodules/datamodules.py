from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl

from .datasets import T2VDataset, TransactionLabelDataset


def fit_discretizer(n_bins: int, train_sequences: pd.DataFrame) -> KBinsDiscretizer:
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
    all_amounts = list()
    for i in range(len(train_sequences)):
        all_amounts += train_sequences.iloc[i]
    discretizer.fit(np.array(all_amounts).reshape(-1, 1))
    return discretizer


class TransactionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_size: int,
        batch_size: int,
        train_sequences: pd.DataFrame,
        val_sequences: pd.DataFrame,
        mcc2id: Dict[int, int],
        discretizer_bins: int
    ):
        super().__init__()
        self.window_size = window_size
        self.batch_size  = batch_size

        discretizer = fit_discretizer(discretizer_bins, train_sequences[1])

        mcc_codes, amnts = train_sequences
        mcc_seqs =  [torch.LongTensor([mcc2id[code] for code in sequence]) for sequence in mcc_codes]
        amnt_seqs = [torch.LongTensor(discretizer.transform(np.array(sequence).reshape(-1, 1))).view(-1) + 1 for sequence in amnts]
        self.train_ds = T2VDataset(mcc_seqs, amnt_seqs, mcc2id, self.window_size)

        mcc_codes, amnts = val_sequences
        mcc_seqs =  [torch.LongTensor([mcc2id[code] for code in sequence]) for sequence in mcc_codes]
        amnt_seqs = [torch.LongTensor(discretizer.transform(np.array(sequence).reshape(-1, 1))).view(-1) + 1 for sequence in amnts]
        self.val_ds = T2VDataset(mcc_seqs, amnt_seqs, mcc2id, self.window_size)
    

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._tr2vec_collate
        )
    

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self._tr2vec_collate
        )
    
    @staticmethod
    def _tr2vec_collate(batch: torch.Tensor) -> Tuple[
    torch.LongTensor,
    torch.Tensor,
    torch.LongTensor,
    torch.Tensor,
    torch.LongTensor
    ]:
        ctx_mccs, ctx_amnts, center_mccs, center_amnts, ctx_lengths = zip(*batch)
        ctx_mccs = pad_sequence(ctx_mccs, batch_first=True, padding_value=0)
        ctx_amnts = pad_sequence(ctx_amnts, batch_first=True, padding_value=0)
        ctx_lengths = torch.LongTensor(ctx_lengths)
        center_mccs = torch.LongTensor(center_mccs)
        center_amnts = torch.LongTensor(center_amnts)
        return ctx_mccs, ctx_amnts, center_mccs, center_amnts, ctx_lengths
    

class TransactionRNNDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        train_sequences: pd.DataFrame,
        val_sequences: pd.DataFrame,
        test_sequences: pd.DataFrame,
        mcc2id: Dict[int, int],
        discretizer_bins: int
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        discretizer = fit_discretizer(discretizer_bins, train_sequences)
        
        mcc_seqs    = train_sequences.small_group
        amnt_seqs   = train_sequences.amount_rur
        labels      = train_sequences.target_flag.tolist()
        mcc_seqs  = [torch.LongTensor([mcc2id[code] for code in seq]) for seq in mcc_seqs]
        amnt_seqs = [torch.LongTensor(discretizer.transform(np.array(seq).reshape(-1, 1))).view(-1) + 1 for seq in amnt_seqs]
        self.train_ds = TransactionLabelDataset(mcc_seqs, amnt_seqs, labels)

        mcc_seqs    = val_sequences.small_group
        amnt_seqs   = val_sequences.amount_rur
        labels      = val_sequences.target_flag.tolist()
        mcc_seqs  = [torch.LongTensor([mcc2id[code] for code in seq]) for seq in mcc_seqs]
        amnt_seqs = [torch.LongTensor(discretizer.transform(np.array(seq).reshape(-1, 1))).view(-1) + 1 for seq in amnt_seqs]
        self.val_ds = TransactionLabelDataset(mcc_seqs, amnt_seqs, labels)

        mcc_seqs    = test_sequences.small_group
        amnt_seqs   = test_sequences.amount_rur
        labels      = test_sequences.target_flag.tolist()
        mcc_seqs  = [torch.LongTensor([mcc2id[code] for code in seq]) for seq in mcc_seqs]
        amnt_seqs = [torch.LongTensor(discretizer.transform(np.array(seq).reshape(-1, 1))).view(-1) + 1 for seq in amnt_seqs]
        self.test_ds = TransactionLabelDataset(mcc_seqs, amnt_seqs, labels)

    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._rnn_collate
        )
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self._rnn_collate
        )
    

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=self._rnn_collate
        )


    def _rnn_collate(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mcc_seqs, amnt_seqs, labels = zip(*batch)
        lengths = torch.LongTensor([len(seq) for seq in mcc_seqs])
        mcc_seqs = pad_sequence(mcc_seqs, batch_first=True)
        amnt_seqs = pad_sequence(amnt_seqs, batch_first=True)
        labels = torch.LongTensor(labels)
        return mcc_seqs, amnt_seqs, labels, lengths
