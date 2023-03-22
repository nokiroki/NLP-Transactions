import os
from typing import Tuple

import numpy as np
import pandas as pd

from utils.data_utils import global_context, split_data, weekends
from utils.config_utils import DataConf, ModelConf, ClassificationParamsConf


def rb_preprocessing(
    data_conf: DataConf,
    model_conf: ModelConf,
    params_conf: ClassificationParamsConf
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(os.path.join(data_conf.data_dir, 'rosbank', 'train.csv'))
    transactions['TRDATETIME'] = pd.to_datetime(transactions['TRDATETIME'], format=r'%d%b%y:%H:%M:%S')
    transactions = transactions.sort_values(by=['TRDATETIME'])
    transactions['hour'] = transactions.TRDATETIME.dt.hour
    transactions['day'] = transactions.TRDATETIME.dt.day
    transactions['day_of_week'] = transactions.TRDATETIME.dt.day_of_week
    transactions['month'] = transactions.TRDATETIME.dt.month
    transactions = transactions.rename(columns={'cl_id':'client_id', 'MCC':'small_group', 'amount':'amount_rur'})

    mcc2id = dict(zip(
        transactions.small_group.unique(), 
        np.arange(transactions.small_group.nunique()) + 1
    ))

    transactions['small_group'] = transactions['small_group'].map(mcc2id)

    sequences = transactions.groupby('client_id').agg({
        'small_group': lambda x: x.tolist(),
        'amount_rur':  lambda x: x.tolist(),
        'hour':        lambda x: x.tolist(),
        'day':         lambda x: x.tolist(),
        'day_of_week': lambda x: x.tolist(),
        'month':       lambda x: x.tolist(),
        'target_flag': lambda x: x.tolist()[0],
    })
    if params_conf.use_global_features:
        transactions = global_context(transactions, params_conf.global_features_step)
        sequences = pd.concat((sequences, transactions.groupby('client_id').agg({
            'average_amt':  lambda x: x.tolist(),
            'top_mcc_1':    lambda x: x.tolist(),
            'top_mcc_2':    lambda x: x.tolist(),
            'top_mcc_3':    lambda x: x.tolist()
        })), axis=1)

    if params_conf.is_weekends:
        transactions = weekends(transactions)
        sequences = pd.concat((sequences, transactions.groupby('client_id').agg({
            'weekends': lambda x: x.tolist()
        })), axis=1)

    # Если задача mcc, то вручную считаем output_dim
    if model_conf.task == 'mcc_classification':
        params_conf.output_dim = transactions.small_group.nunique()
        new_seqs = sequences.drop(columns='target_flag', axis=1).apply(
            lambda row: [l[:-1] for l in row],
            axis=0
        )
        new_seqs['target_flag'] = sequences[['small_group']].apply(
            lambda mcc: mcc[0][-1],
            axis=1
        )
        sequences = new_seqs

        mask = sequences['small_group'].apply(lambda x: len(x)) > 5
        sequences = sequences[mask]

    return split_data(sequences, val_size=.2)
