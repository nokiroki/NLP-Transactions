from typing import Dict, Tuple, List, Optional, Iterable, Any

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
        discretizer_bins: int,
        num_workers: int = 1
    ):
        super().__init__()
        self.window_size = window_size
        self.batch_size  = batch_size
        self.num_workers = num_workers

        discretizer = fit_discretizer(discretizer_bins, train_sequences[1])

        mcc_codes, amnts = train_sequences
        mcc_seqs =  [torch.LongTensor(sequence) for sequence in mcc_codes]
        amnt_seqs = [torch.LongTensor(discretizer.transform(np.array(sequence).reshape(-1, 1))).view(-1) + 1 for sequence in amnts]
        self.train_ds = T2VDataset(mcc_seqs, amnt_seqs, self.window_size)

        mcc_codes, amnts = val_sequences
        mcc_seqs =  [torch.LongTensor(sequence) for sequence in mcc_codes]
        amnt_seqs = [torch.LongTensor(discretizer.transform(np.array(sequence).reshape(-1, 1))).view(-1) + 1 for sequence in amnts]
        self.val_ds = T2VDataset(mcc_seqs, amnt_seqs, self.window_size)
    

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
        discretizer_bins: int,
        is_global_features: bool = True,
        m_last: int = 100,
        m_period: int = 0,
        period: str = 'day',
        num_workers: int = 1
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.is_global_features = is_global_features
        self.m_last = m_last
        self.m_period = m_period
        self.period = period

        self.discretizer = fit_discretizer(discretizer_bins, train_sequences['amount_rur'])
        
        self.train_ds   = self.create_dataset(train_sequences)
        self.val_ds     = self.create_dataset(val_sequences)
        self.test_ds    = self.create_dataset(test_sequences)


    def create_dataset(self, sequences: pd.DataFrame) -> TransactionLabelDataset:
        mcc_seqs      = sequences.small_group
        amnt_seqs     = sequences.amount_rur
        period_seqs   = sequences[self.period]
        labels        = sequences.target_flag.tolist()

        mcc_seqs_processed, amnt_seqs_processed = self.process_sequences(mcc_seqs, amnt_seqs, period_seqs)

        if self.is_global_features:
            avg_amnt, top_mcc = self.get_agg_func(sequences)
            return TransactionLabelDataset(mcc_seqs_processed, amnt_seqs_processed, labels, avg_amnt, top_mcc)
        else:
            return TransactionLabelDataset(mcc_seqs_processed, amnt_seqs_processed, labels)


    def process_sequences(
        self,
        mcc_seqs: Iterable[int],
        amnt_seqs: Iterable[int],
        period_seqs: Iterable[Any]
    ):
        mcc_seqs_processed, amnt_seqs_processed = [], []
        for mcc_seq, amnt_seq, period_seq in zip(mcc_seqs, amnt_seqs, period_seqs):
            cur_user        = pd.DataFrame({'mcc': mcc_seq, 'amnt': amnt_seq, 'period': period_seq})
            cur_user_subset = cur_user.iloc[-self.m_last:, :].copy()
            if (len(cur_user) > self.m_last) and (self.m_period > 0):
                cur_user_first         = cur_user.iloc[:-self.m_last, :].copy()
                last_period            = cur_user_subset['period'].iloc[-1]
                cur_user_period_subset = cur_user_first[cur_user_first['period'] == last_period].iloc[-self.m_period:, :].copy()
                cur_user_subset        = pd.concat([cur_user_period_subset, cur_user_subset], axis=0)
                
            mcc_seqs_processed.append(torch.LongTensor(cur_user_subset['mcc']))
            amnt_seqs_processed.append(torch.LongTensor(self.discretizer.transform(np.array(cur_user_subset['amnt']).reshape(-1, 1))).view(-1) + 1)

        return mcc_seqs_processed, amnt_seqs_processed

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self._rnn_collate_with_gc if self.is_global_features else self._rnn_collate
        )
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._rnn_collate_with_gc if self.is_global_features else self._rnn_collate
        )
    

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._rnn_collate_with_gc if self.is_global_features else self._rnn_collate
        )


    @staticmethod
    def _rnn_collate_with_gc(batch: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        mcc_seqs, amnt_seqs, labels, avg_amnt, top_mcc = zip(*batch)
        lengths = torch.LongTensor([len(seq) for seq in mcc_seqs])
        mcc_seqs = pad_sequence(mcc_seqs, batch_first=True)
        amnt_seqs = pad_sequence(amnt_seqs, batch_first=True)
        avg_amnt = pad_sequence(avg_amnt, batch_first=True)
        top_mcc = pad_sequence(top_mcc, batch_first=True)
        labels = torch.LongTensor(labels)
        return mcc_seqs, amnt_seqs, labels, lengths, avg_amnt, top_mcc
    
    @staticmethod
    def _rnn_collate(batch: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        mcc_seqs, amnt_seqs, labels, _, _ = zip(*batch)
        lengths = torch.LongTensor([len(seq) for seq in mcc_seqs])
        mcc_seqs = pad_sequence(mcc_seqs, batch_first=True)
        amnt_seqs = pad_sequence(amnt_seqs, batch_first=True)
        labels = torch.LongTensor(labels)
        return mcc_seqs, amnt_seqs, labels, lengths, None, None
    

    def get_agg_func(self, sequences) -> Tuple[torch.Tensor, torch.Tensor]:        
        avg_amt = sequences.average_amt
        top_mcc_1 = sequences.top_mcc_1
        top_mcc_2 = sequences.top_mcc_2
        top_mcc_3 = sequences.top_mcc_3

        avg_amt = [torch.LongTensor(self.discretizer.transform(np.array(seq).reshape(-1, 1))).view(-1) + 1 for seq in avg_amt]
        top_mcc_seqs = [torch.stack((
                torch.LongTensor(seq_1),
                torch.LongTensor(seq_2),
                torch.LongTensor(seq_3)
            ), -1) for seq_1, seq_2, seq_3 in zip(top_mcc_1, top_mcc_2, top_mcc_3)
        ]

        return avg_amt, top_mcc_seqs
