import os

import numpy as np
import pandas as pd

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.churn_classification import TransactionGRU, Transformer
from models.callbacks import FreezeEmbeddings
from datamodules import TransactionRNNDataModule
from utils.data_utils import split_data, global_context
from utils.config_utils import get_config_with_dirs


if __name__ == '__main__':
    config, (data_dir, logging_dir) = get_config_with_dirs('config.ini')
    
    # Чтение данных из конфига
    model_type = config.get('Churn_classification', 'model_type')
    conf_section = 'RNN' if model_type == 'rnn' else 'Transformer'

    batch_size          = config.getint(conf_section, 'batch_size')
    lr                  = config.getfloat(conf_section, 'lr')
    epochs              = config.getint(conf_section, 'epochs')
    emb_type            = config.get(conf_section, 'emb_type')
    mcc_vocab_size      = config.getint(conf_section, 'mcc_vocab_size')
    mcc_embed_size      = config.getint(conf_section, 'mcc_emb_size')
    amnt_bins           = config.getint(conf_section, 'amnt_bins')
    amnt_emb_size       = config.getint(conf_section, 'amnt_emb_size')
    emb_size            = config.getint(conf_section, 'emb_size')
    layers              = config.getint(conf_section, 'layers')
    hidden_dim          = config.getint(conf_section, 'hidden_dim')
    dropout             = config.getfloat(conf_section, 'dropout')
    permutation         = config.getboolean(conf_section, 'permutation')
    pe                  = config.getboolean(conf_section, 'pe')
    use_global_features = config.getboolean('All_models', 'use_global_features')
    num_workers         = config.getint('All_models', 'num_workers')
    num_heads           = config.getint(conf_section, 'n_heads') if model_type == 'transformer' else None
    
    # Чтения файла росбанка
    transactions = pd.read_csv(os.path.join(data_dir, 'rosbank\\train.csv'))
    transactions['TRDATETIME'] = pd.to_datetime(transactions['TRDATETIME'], format=r'%d%b%y:%H:%M:%S')
    transactions = transactions.sort_values(by=['TRDATETIME'])
    transactions = transactions.rename(columns={'cl_id':'client_id', 'MCC':'small_group', 'amount':'amount_rur'})

    sequences = transactions.groupby('client_id').agg({
        'small_group': lambda x: x.tolist(),
        'amount_rur': lambda x: x.tolist(),
        'target_flag': lambda x: x.tolist()[0],
    })
    if use_global_features:
        transactions = global_context(transactions)
        sequences = pd.concat((sequences, transactions.groupby('client_id').agg({
            'average_amt': lambda x: x.tolist(),
            'top_mcc_1': lambda x: x.tolist(),
            'top_mcc_2': lambda x: x.tolist(),
            'top_mcc_3': lambda x: x.tolist()
        })), axis=1)


    train_sequences, val_sequences, test_sequences = split_data(sequences, use_train_ratio=1.)
    mcc2id = dict(zip(
        transactions.small_group.unique(), 
        np.arange(transactions.small_group.nunique()) + 1
    ))


    results = list()
    #TODO сделать через интерфейс подгрузку весов
    weights = torch.load(os.path.join(
        logging_dir,
        'best_model_params',
        ''.join((
            'tr2vec_mcc=16',
            '_amnt=8',
            '_emb=16',
            '_window=10',
            '_loss=ordinal.pth'
        ))
    ))
    
    # Цикл обучения для оценки uncertainty
    for _ in range(5):
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
            pe
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
            pe   
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
            num_workers=num_workers
        )

        early_stop_callback = EarlyStopping(
            monitor='val_auroc',
            min_delta=1e-3,
            patience=4,
            verbose=False,
            mode='max'
        )

        checkpoint = ModelCheckpoint(monitor='val_auroc', mode='max')
        
        callbacks = [checkpoint, early_stop_callback, FreezeEmbeddings()]

        tb_logger = TensorBoardLogger(os.path.join(logging_dir, 'tb_logs\\transformer'), 'data_without_global_context')

        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            log_every_n_steps=20,
            logger=tb_logger,
            deterministic=True,
            callbacks=callbacks,
            max_epochs=epochs,
            auto_lr_find=True
        )

        trainer.fit(model, datamodule=datamodule)
        model_best = TransactionGRU.load_from_checkpoint(checkpoint.best_model_path) \
                    if model_type == 'rnn' else \
                    Transformer.load_from_checkpoint(checkpoint.best_model_path)
        res = trainer.test(model_best, datamodule)[0]['test_auroc']
        results.append(res)
