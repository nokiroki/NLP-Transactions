import os

import pandas as pd

from utils.data_utils import global_context


def rb_preprocessing(
    data_dir: str,
    use_global_features: bool,
    global_features_step: int,
    is_weekend: bool
):
    transactions = pd.read_csv(os.path.join(data_dir, 'rosbank', 'train.csv'))
    transactions['TRDATETIME'] = pd.to_datetime(transactions['TRDATETIME'], format=r'%d%b%y:%H:%M:%S')
    transactions = transactions.sort_values(by=['TRDATETIME'])
    transactions['hour'] = transactions.TRDATETIME.dt.hour
    transactions['day'] = transactions.TRDATETIME.dt.day
    transactions['day_of_week'] = transactions.TRDATETIME.dt.day_of_week
    transactions['month'] = transactions.TRDATETIME.dt.month
    transactions = transactions.rename(columns={'cl_id':'client_id', 'MCC':'small_group', 'amount':'amount_rur'})

    sequences = transactions.groupby('client_id').agg({
        'small_group': lambda x: x.tolist(),
        'amount_rur':  lambda x: x.tolist(),
        'hour':        lambda x: x.tolist(), 
        'day':         lambda x: x.tolist(), 
        'day_of_week': lambda x: x.tolist(), 
        'month':       lambda x: x.tolist(), 
        'target_flag': lambda x: x.tolist()[0],
    })
    if use_global_features:
        transactions = global_context(transactions, global_features_step)
        sequences = pd.concat((sequences, transactions.groupby('client_id').agg({
            'average_amt': lambda x: x.tolist(),
            'top_mcc_1': lambda x: x.tolist(),
            'top_mcc_2': lambda x: x.tolist(),
            'top_mcc_3': lambda x: x.tolist()
        })), axis=1)
