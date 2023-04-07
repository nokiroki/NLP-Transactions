import os
from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn

from tqdm.auto import tqdm

from sklearn.preprocessing import KBinsDiscretizer

from utils.data_utils import global_context, split_data, weekends, global_context_emb_avg
from utils.config_utils import DataConf, ModelConf, ClassificationParamsConf


def rb_preprocessing(
    data_conf: DataConf,
    model_conf: ModelConf,
    params_conf: ClassificationParamsConf
) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
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
    discretizer = KBinsDiscretizer(params_conf.amnt_bins, encode='ordinal', subsample=int(2e5))

    transactions['small_group'] = transactions['small_group'].map(mcc2id)
    transactions['amount_rur_kb'] = discretizer.fit_transform(
        transactions['amount_rur'].values.reshape(-1, 1)
    ).astype(np.int32) + 1

    sequences = transactions.groupby('client_id').agg({
        'small_group':      lambda x: x.tolist(),
        'amount_rur_kb':    lambda x: x.tolist(),
        'hour':             lambda x: x.tolist(),
        'day':              lambda x: x.tolist(),
        'day_of_week':      lambda x: x.tolist(),
        'month':            lambda x: x.tolist(),
        'target_flag':      lambda x: x.tolist()[0],
    })
    if params_conf.use_global_features:
        transactions = global_context(transactions, params_conf.global_features_step)
        transactions['average_amt'] = discretizer.transform(
            transactions['average_amt'].values.reshape(-1, 1)
        ).astype(np.int32) + 1
        sequences = pd.concat((sequences, transactions.groupby('client_id').agg({
            'average_amt':  lambda x: x.tolist(),
            'top_mcc_1':    lambda x: x.tolist(),
            'top_mcc_2':    lambda x: x.tolist(),
            'top_mcc_3':    lambda x: x.tolist(),
            'gc_id':        lambda x: x.tolist()
        })), axis=1)
    sequences = sequences.rename(columns={'amount_rur_kb': 'amount_rur'})
    transactions = transactions.drop(columns=['amount_rur'], axis=1)
    transactions = transactions.rename(columns={'amount_rur_kb': 'amount_rur'})

    if params_conf.use_global_features and params_conf.global_feature_type == 1:
        weights = torch.load(os.path.join(
            data_conf.emb_dir,
            'embedding_weights',
            model_conf.emb_weights_name
            ))
        mcc_embeddings = nn.Embedding(
            params_conf.mcc_vocab_size + 1,
            params_conf.mcc_embed_size,
            padding_idx=0
        )
        amnt_embeddings = nn.Embedding(
            params_conf.amnt_bins + 1,
            params_conf.amnt_emb_size,
            padding_idx=0
        )

        mcc_embeddings.weight.data = weights['mccs']
        amnt_embeddings.weight.data = weights['amnts']

        mcc_embeddings = mcc_embeddings.to(model_conf.device)
        amnt_embeddings = amnt_embeddings.to(model_conf.device)

        gc_emb_amnt, gc_emb_mcc = global_context_emb_avg(
            transactions,
            amnt_embeddings,
            mcc_embeddings,
            device=model_conf.device
        )

        avg_amnt_seqs = []
        avg_mcc_seqs = []
        for seq in tqdm(sequences.iloc, total=sequences.shape[0]):
            amnt_seq = []
            mcc_seq = []
            
            for i in seq['gc_id']:
                amnt_seq.append(gc_emb_amnt[i])
                mcc_seq.append(gc_emb_mcc[i])
            avg_amnt_seqs.append(amnt_seq)
            avg_mcc_seqs.append(mcc_seq)
        sequences['amnt_avg_embed'] = avg_amnt_seqs
        sequences['mcc_avg_embed'] = avg_mcc_seqs


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

    return transactions, split_data(sequences, val_size=.2)
