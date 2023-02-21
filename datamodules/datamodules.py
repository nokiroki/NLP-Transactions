from typing import Dict, Tuple, List

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
        discretizer_bins: int,
        num_workers: int = 1
    ):
        super().__init__()
        self.window_size = window_size
        self.batch_size  = batch_size
        self.num_workers = num_workers

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
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self._tr2vec_collate
        )
    

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
        discretizer_bins: int,
        is_global_features: bool = True,
        num_workers: int = 1
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.mcc2id = mcc2id

        self.discretizer = fit_discretizer(discretizer_bins, train_sequences['amount_rur'])
        
        mcc_seqs_train      = train_sequences.small_group
        amnt_seqs_train     = train_sequences.amount_rur
        labels_train        = train_sequences.target_flag.tolist()
        mcc_seqs_train      = [torch.LongTensor([mcc2id[code] for code in seq]) for seq in mcc_seqs_train]
        amnt_seqs_train     = [torch.LongTensor(self.discretizer.transform(np.array(seq).reshape(-1, 1))).view(-1) + 1 for seq in amnt_seqs_train]

        mcc_seqs_val        = val_sequences.small_group
        amnt_seqs_val       = val_sequences.amount_rur
        labels_val          = val_sequences.target_flag.tolist()
        mcc_seqs_val        = [torch.LongTensor([mcc2id[code] for code in seq]) for seq in mcc_seqs_val]
        amnt_seqs_val       = [torch.LongTensor(self.discretizer.transform(np.array(seq).reshape(-1, 1))).view(-1) + 1 for seq in amnt_seqs_val]

        mcc_seqs_test       = test_sequences.small_group
        amnt_seqs_test      = test_sequences.amount_rur
        labels_test         = test_sequences.target_flag.tolist()
        mcc_seqs_test       = [torch.LongTensor([mcc2id[code] for code in seq]) for seq in mcc_seqs_test]
        amnt_seqs_test      = [torch.LongTensor(self.discretizer.transform(np.array(seq).reshape(-1, 1))).view(-1) + 1 for seq in amnt_seqs_test]

        if is_global_features:
            (avg_amnt_train, top_mcc_train), \
            (avg_amnt_val, top_mcc_val), \
            (avg_amnt_test, top_mcc_test) = self._get_agg_func()

            self.train_ds   = TransactionLabelDataset(mcc_seqs_train, amnt_seqs_train, labels_train, avg_amnt_train, top_mcc_train)
            self.val_ds     = TransactionLabelDataset(mcc_seqs_val, amnt_seqs_val, labels_val, avg_amnt_val, top_mcc_val)
            self.test_ds    = TransactionLabelDataset(mcc_seqs_test, amnt_seqs_test, labels_test, avg_amnt_test, top_mcc_test)
        else:
            self.train_ds   = TransactionLabelDataset(mcc_seqs_train, amnt_seqs_train, labels_train)
            self.val_ds     = TransactionLabelDataset(mcc_seqs_val, amnt_seqs_val, labels_val)
            self.test_ds    = TransactionLabelDataset(mcc_seqs_test, amnt_seqs_test, labels_test)

    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self._rnn_collate
        )
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._rnn_collate
        )
    

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._rnn_collate
        )


    @staticmethod
    def _rnn_collate(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mcc_seqs, amnt_seqs, labels, avg_amnt, top_mcc = zip(*batch)
        lengths = torch.LongTensor([len(seq) for seq in mcc_seqs])
        mcc_seqs = pad_sequence(mcc_seqs, batch_first=True)
        amnt_seqs = pad_sequence(amnt_seqs, batch_first=True)
        avg_amnt = pad_sequence(avg_amnt, batch_first=True)
        top_mcc = pad_sequence(top_mcc, batch_first=True)
        labels = torch.LongTensor(labels)
        return mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc
    

    def _get_agg_func(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        train_val_test_agg_features = list()
        for sequences in (self.train_sequences, self.val_sequences, self.test_sequences):
            avg_amt = sequences.average_amt
            top_mcc_1 = sequences.top_mcc_1
            top_mcc_2 = sequences.top_mcc_2
            top_mcc_3 = sequences.top_mcc_3

            avg_amt = [torch.LongTensor(self.discretizer.transform(np.array(seq).reshape(-1, 1))).view(-1) + 1 for seq in avg_amt]
            top_mcc_seqs = [torch.stack((
                    torch.LongTensor([self.mcc2id[code] for code in seq_1]),
                    torch.LongTensor([self.mcc2id[code] for code in seq_2]),
                    torch.LongTensor([self.mcc2id[code] for code in seq_3])
                ), -1) for seq_1, seq_2, seq_3 in zip(top_mcc_1, top_mcc_2, top_mcc_3)
            ]

            train_val_test_agg_features.append((avg_amt, top_mcc_seqs))
        return train_val_test_agg_features
