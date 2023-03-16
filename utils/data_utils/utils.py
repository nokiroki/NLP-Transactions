from typing import Tuple, List
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm


def split_data(
    data: pd.DataFrame,
    train_size: float = .8,
    use_train_ratio:float = 1.,
    val_size: float = .5,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)

    train_ids = np.random.choice(len(data), int(len(data) * train_size), replace=False)
    other_ids = np.setdiff1d(np.arange(len(data)), train_ids)
    train_ids = np.random.choice(train_ids, int(len(train_ids) * use_train_ratio), replace=False)
    train_data = data.iloc[train_ids]
    other_data = data.iloc[other_ids]

    val_size = 0.5
    val_ids = np.random.choice(len(other_data), int(len(other_data) * val_size))
    test_ids = np.setdiff1d(np.arange(len(other_data)), val_ids)

    val_data = other_data.iloc[val_ids]
    test_data = other_data.iloc[test_ids]


    return train_data, val_data, test_data


def global_context(
    transactions: pd.DataFrame,
    time_step: int = 1,
    datetime_column: str = 'TRDATETIME',
    tr_amount_column: str = 'amount_rur',
    tr_mcc_code_column: str = 'small_group'
) -> pd.DataFrame:
    start_time = transactions.iloc[0]['TRDATETIME']
    start_index = 0

    mean_tr = np.zeros(transactions.shape[0])
    top_mcc = np.zeros((transactions.shape[0], 3))

    for i in tqdm(range(transactions.shape[0])):
        curr_time = transactions.iloc[i][datetime_column]
        if transactions.iloc[i][datetime_column] > (start_time + timedelta(days=time_step)) or i == (transactions.shape[0] - 1):
            subsample = transactions.iloc[start_index:i]
            mean_tr_smpl = subsample[tr_amount_column].mean()
            top_mcc_smpl = subsample.groupby(tr_mcc_code_column)[tr_mcc_code_column].count().sort_values(ascending=False).iloc[:3].index
            if len(top_mcc_smpl) < 3:
                border_index = 3 - len(top_mcc_smpl)
                top_mcc_smpl = np.array(top_mcc_smpl)
                top_mcc_smpl.resize(3)
                top_mcc_smpl[-border_index:] = top_mcc_smpl[-(1 + border_index)]

            mean_tr[start_index:i] = mean_tr_smpl
            top_mcc[start_index:i] = top_mcc_smpl

            start_time = curr_time
            start_index = i

    transactions_new = transactions.copy()

    transactions_new['average_amt'] = mean_tr
    transactions_new[['top_mcc_1', 'top_mcc_2', 'top_mcc_3']] = top_mcc.astype(np.int32)

    transactions_new.drop(transactions_new[transactions_new['average_amt'] == 0].index, axis=0, inplace=True)

    return transactions_new


def weekends(transactions: pd.DataFrame, datetime_column: str = 'TRDATETIME') -> pd.DataFrame:

    transactions_new = transactions.copy()
    transactions_new['is_weekend'] = transactions_new[[datetime_column]].apply(
        lambda x: 1 if x[0].weekday() >= 5 else 0,
        axis=1
    )
    return transactions_new


def mask_drop_save_tockens(
    batch: torch.Tensor,
    random_mask: torch.Tensor,
    save_tockens: List[int]
) -> torch.Tensor:

    rand_mask = rand_mask.copy()
    for token in save_tockens:
        rand_mask *= (batch != token)

    return rand_mask


def masking_one_batch(
    batch: torch.Tensor,
    random_mask: torch.Tensor,
    mask_token: int,
    save_tockens: List[int]

) -> Tuple[torch.Tensor, torch.Tensor]:

    mask_for_batch = mask_drop_save_tockens(batch, random_mask, save_tockens)  
    new_bacth = batch.mask_fill_(mask_for_batch, mask_token)  

    return new_bacth, mask_for_batch



def masking_all_batches(
    list_changing_batches: List[torch.Tensor],
    mask_tokens: List[int],
    masked_rate: float = 0.15,
    save_tokens: List[List[int]] = [[0], [0]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:


    rand_value = torch.rand(list_changing_batches[0].shape)
    rand_mask = (rand_value < masked_rate) 

    rand_masks_per_batch = []
    new_batches = []

    for idx_batch, batch in enumerate(list_changing_batches):

        new_batch, mask_for_batch = masking_one_batch(batch, rand_mask, mask_tokens[idx_batch], save_tokens[idx_batch])

        rand_masks_per_batch.append(mask_for_batch)
        new_batches.append(new_batch)

    return new_batches, rand_masks_per_batch

