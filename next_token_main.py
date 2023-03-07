import os

import numpy as np
import pandas as pd

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.next_token_classification import TransactionGRU, Transformer
from models.callbacks import FreezeEmbeddings, UnfreezeEmbeddings
from datamodules import TransactionRNNDataModule
from utils.data_utils import split_data, global_context
from utils.config_utils import get_config_with_dirs_old


if __name__ == '__main__':
    config, (data_dir, logging_dir, emb_dir) = get_config_with_dirs_old('config.ini')
    
    # Чтение данных из конфига
    model_type = config.get('All_models', 'model_type').lower()
    conf_section    = 'RNN' if model_type == 'rnn' else 'Transformer'
    experiment_name         = config.get('All_models', 'experiment_name')
    emb_weigths_name        = config.get('All_models', 'emb_weigths_name')
    train_embeddings        = config.getboolean('All_models', 'train_embedding')

    batch_size              = config.getint(conf_section, 'batch_size')
    lr                      = config.getfloat(conf_section, 'lr')
    epochs                  = config.getint(conf_section, 'epochs')
    num_workers             = config.getint('All_models', 'num_workers')
    n_experiments           = config.getint('All_models', 'n_experiments')

    emb_type                = config.get(conf_section, 'emb_type')
    mcc_vocab_size          = config.getint(conf_section, 'mcc_vocab_size')
    mcc_embed_size          = config.getint(conf_section, 'mcc_emb_size')
    amnt_bins               = config.getint(conf_section, 'amnt_bins')
    amnt_emb_size           = config.getint(conf_section, 'amnt_emb_size')
    emb_size                = config.getint(conf_section, 'emb_size')
    layers                  = config.getint(conf_section, 'layers')
    hidden_dim              = config.getint(conf_section, 'hidden_dim')
    dropout                 = config.getfloat(conf_section, 'dropout')
    permutation             = config.getboolean(conf_section, 'permutation')
    pe                      = config.getboolean(conf_section, 'pe')
    use_global_features     = config.getboolean('All_models', 'use_global_features')
    global_features_step    = config.getint('All_models', 'global_features_step')
    m_last                  = config.getint('All_models', 'm_last')
    m_period                = config.getint('All_models', 'm_period')
    period                  = config.get('All_models', 'period')
    num_heads               = config.getint(conf_section, 'n_heads') if model_type == 'transformer' else None

    # Необходим файл с токеном для логгирование на comet
    with open('api_token.txt') as f:
        api_token = f.read()
    
    # Чтения файла росбанка
    transactions = pd.read_csv(os.path.join(data_dir, 'rosbank', 'train.csv'))
    transactions['TRDATETIME'] = pd.to_datetime(transactions['TRDATETIME'], format=r'%d%b%y:%H:%M:%S')
    transactions = transactions.sort_values(by=['TRDATETIME'])
    transactions['hour'] = transactions.TRDATETIME.dt.hour
    transactions['day'] = transactions.TRDATETIME.dt.day
    transactions['day_of_week'] = transactions.TRDATETIME.dt.day_of_week
    transactions['month'] = transactions.TRDATETIME.dt.month
    transactions = transactions.rename(columns={'cl_id':'client_id', 'MCC':'small_group', 'amount':'amount_rur'})

    n_classes = transactions.small_group.nunique()
    mcc2id = dict(zip(
        transactions.small_group.unique(), 
        np.arange(transactions.small_group.nunique()) + 1
    ))


    sequences = transactions.groupby('client_id').agg({
    'small_group': lambda x: x.tolist()[:-1],
    'amount_rur':  lambda x: x.tolist()[:-1],
    'hour':        lambda x: x.tolist()[:-1], 
    'day':         lambda x: x.tolist()[:-1], 
    'day_of_week': lambda x: x.tolist()[:-1], 
    'month':       lambda x: x.tolist()[:-1], 
    })

    label = transactions.groupby('client_id').agg({
        'small_group': lambda x: x.tolist()[-1]
    }).rename(columns={'small_group':'target_flag'})

    sequences = sequences.join(label)
    mask = sequences['small_group'].apply(lambda x: len(x)) > 5
    sequences = sequences[mask]
    sequences['target_flag'] = sequences['target_flag'].map(mcc2id)


    if use_global_features:
        transactions = global_context(transactions, global_features_step)
        sequences = pd.concat((sequences, transactions.groupby('client_id').agg({
            'average_amt': lambda x: x.tolist(),
            'top_mcc_1': lambda x: x.tolist(),
            'top_mcc_2': lambda x: x.tolist(),
            'top_mcc_3': lambda x: x.tolist()
        })), axis=1)


    train_sequences, val_sequences, test_sequences = split_data(sequences, use_train_ratio=1.)

    results = list()
    #TODO сделать через интерфейс подгрузку весов
    weights = torch.load(os.path.join(
        emb_dir,
        'embedding_weights',
        emb_weigths_name
    ))
    
    # Цикл обучения для оценки uncertainty
    for i in range(n_experiments):
        model = TransactionGRU(
            emb_type,
            mcc_vocab_size,
            mcc_embed_size,
            amnt_bins,
            amnt_emb_size,
            emb_size,
            hidden_dim,
            layers,
            dropout,
            lr,
            permutation,
            pe,
            n_classes
        ) if model_type == 'rnn' else \
        Transformer(
            emb_type,
            mcc_vocab_size,
            mcc_embed_size,
            amnt_bins,
            amnt_emb_size,
            emb_size,
            hidden_dim,
            layers,
            dropout,
            lr,
            num_heads,
            permutation,
            pe,
            n_classes  
        )

        model.set_embeddings(weights['mccs'], weights['amnts'])

        datamodule = TransactionRNNDataModule(
            batch_size,
            train_sequences,
            val_sequences,
            test_sequences,
            mcc2id,
            amnt_bins,
            is_global_features=use_global_features,
            m_last=m_last,
            m_period=m_period,
            period=period,
            num_workers=num_workers,
        )

        early_stop_callback = EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-3,
            patience=4,
            verbose=False,
            mode='max'
        )

        checkpoint = ModelCheckpoint(
            monitor='val_accuracy',
            mode='max',
            dirpath=os.path.join(logging_dir, 'checkpoints', experiment_name))

        if train_embeddings:
            callbacks = [checkpoint, early_stop_callback, UnfreezeEmbeddings()]
        else:
            callbacks = [checkpoint, early_stop_callback, FreezeEmbeddings()]

        comet_logger = CometLogger(
            api_key=api_token,
            project_name='NLP-transactions',
            experiment_name=experiment_name
        )

        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            log_every_n_steps=20,
            logger=comet_logger,
            deterministic=True,
            callbacks=callbacks,
            max_epochs=epochs,
            auto_lr_find=True
        )

        
        trainer.fit(model, datamodule=datamodule)

        model_best = model.load_from_checkpoint(checkpoint.best_model_path)
        res = trainer.test(model_best, datamodule)[0]['test_accuracy']
        results.append(res)
