from typing import Tuple
from datetime import timedelta

import numpy as np
import pandas as pd

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


def global_context(transactions: pd.DataFrame, time_step: int = 1) -> pd.DataFrame:
    start_time = transactions.iloc[0]['TRDATETIME']
    start_index = 0

    is_weekend_arr = np.zeros(transactions.shape[0])
    mean_tr = np.zeros(transactions.shape[0])
    top_mcc = np.zeros((transactions.shape[0], 3))

    for i in tqdm(range(transactions.shape[0])):
        curr_time = transactions.iloc[i]['TRDATETIME']
        is_weekend_arr[i] = 1 if curr_time.weekday() >= 5 else 0
        if transactions.iloc[i]['TRDATETIME'] > (start_time + timedelta(days=time_step)) or i == (transactions.shape[0] - 1):
            subsample = transactions.iloc[start_index:i]
            mean_tr_smpl = subsample['amount_rur'].mean()
            top_mcc_smpl = subsample.groupby('small_group')['small_group'].count().sort_values(ascending=False).iloc[:3].index
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

    transactions_new['is_weekend'] = is_weekend_arr
    transactions_new['average_amt'] = mean_tr
    transactions_new[['top_mcc_1', 'top_mcc_2', 'top_mcc_3']] = top_mcc.astype(np.int32)

    transactions_new.drop(transactions_new[transactions_new['average_amt'] == 0].index, axis=0, inplace=True)

    return transactions_new
