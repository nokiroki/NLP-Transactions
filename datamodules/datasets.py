from typing import Iterable, Optional, Dict, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class T2VDataset(Dataset):

    def __init__(
        self, 
        mcc_sequences: Iterable[torch.LongTensor],
        amnt_sequences: Iterable[torch.LongTensor],
        mcc2id: Dict[int, int],
        window_size: int,
        subsample: bool = False,
        class_proportions: Optional[list] = None
    ) -> None:
        if subsample:
            assert class_proportions is not None
            class_proportions = np.array(class_proportions)
            keep_probs = (np.sqrt(class_proportions / 0.2) + 1) * 0.2 / class_proportions
        else:
            keep_probs = np.ones(len(mcc2id))

        self.id2seq_id = []
        self.id2offset = []
        self.window_size = window_size

        mcc_sequences = [seq for seq in mcc_sequences if len(seq) > 1]
        lens = [len(seq) for seq in mcc_sequences]
        for seq_id, l in enumerate(lens):
            self.id2seq_id += [seq_id] * l
            self.id2offset += list(range(l))
        amnt_sequences = [seq for seq in amnt_sequences if len(seq) > 1]

        self.mcc_seqs = mcc_sequences
        self.amnt_seqs = amnt_sequences

    
    def __getitem__(self, id: int) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor
    ]:
        seq_id, offset = self.id2seq_id[id], self.id2offset[id]
        mcc_seq, amnt_seq = self.mcc_seqs[seq_id], self.amnt_seqs[seq_id]
        center_mcc, center_amnt = mcc_seq[offset], amnt_seq[offset]
        left, right = max(offset - self.window_size, 0), min(offset + self.window_size, len(mcc_seq))
        ctx_mcc = torch.cat([mcc_seq[left:offset], mcc_seq[offset + 1:right]])
        ctx_amnt = torch.cat([amnt_seq[left:offset], amnt_seq[offset + 1:right]])
        ctx_length = len(ctx_mcc)
        return ctx_mcc, ctx_amnt, center_mcc, center_amnt, ctx_length
    

    def __len__(self) -> int:
        return len(self.id2seq_id)


class TransactionLabelDataset(Dataset):

    def __init__(self, mcc_seqs: pd.DataFrame, amnt_seqs: pd.DataFrame, labels: pd.DataFrame) -> None:
        self.mcc_seqs   = mcc_seqs
        self.amnt_seqs  = amnt_seqs
        self.labels     = labels

    def __getitem__(self, index: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.mcc_seqs[index], self.amnt_seqs[index], self.labels[index]
    
    def __len__(self) -> int:
        return len(self.labels)