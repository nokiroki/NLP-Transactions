from typing import Tuple

import numpy as np
import pandas as pd

import torch


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
