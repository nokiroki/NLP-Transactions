from configparser import ConfigParser
import os

import numpy as np
import pandas as pd

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from utils import split_data
from models import Transaction2VecJoint
from datamodules import TransactionDataModule


if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini', encoding='utf-8')

    data_dir = config.get('Data', 'data_dir')
    logging_dir = config.get('Logging', 'logging_dir')
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)

    # Подгружаем данные росбанка
    transactions = pd.read_csv(os.path.join(data_dir, 'rosbank\\train.csv'))
    transactions = transactions.sort_values(by=['TRDATETIME'])
    transactions = transactions.rename(columns={'cl_id':'client_id', 'MCC':'small_group', 'amount':'amount_rur'})

    # Создаём словарь соответствия MCC кода и присвоенного нами id
    mcc2id = dict(zip(
        transactions.small_group.unique(), 
        np.arange(transactions.small_group.nunique()) + 1
    ))
    class_proportions = transactions.small_group.value_counts() / len(transactions)
    _id2mcc = {v:int(k) for k, v in mcc2id.items()}
    class_proportions = [class_proportions[_id2mcc[id]] for id in range(1, len(mcc2id) + 1)]

    sequences = transactions.groupby('client_id').agg({'small_group': lambda x: x.tolist(), 'amount_rur': lambda x: x.tolist()})
    train_sequences, val_sequences, test_sequences = split_data(sequences)

    # Импортируем необходимые гиперпараметры из конфига

    window_size     = config.getint('Tr2Vec', 'window_size')
    mcc_vocab_size  = config.getint('Tr2Vec', 'mcc_vocab_size')
    mcc_emb_size    = config.getint('Tr2Vec', 'mcc_emb_size')
    amnt_bins       = config.getint('Tr2Vec', 'amnt_bins')
    amnt_emb_size   = config.getint('Tr2Vec', 'amnt_emb_size')
    emb_size        = config.getint('Tr2Vec', 'emb_size')
    amnt_loss       = config.get('Tr2Vec', 'amnt_loss')
    lr              = config.getfloat('Tr2Vec', 'lr')
    batch_size      = config.getint('Tr2Vec', 'batch_size')
    epochs          = config.getint('Tr2Vec', 'epochs')

    # Датамодуль
    datamodule = TransactionDataModule(
        window_size,
        batch_size,
        (train_sequences.small_group, train_sequences.amount_rur),
        (val_sequences.small_group, val_sequences.amount_rur),
        mcc2id,
        amnt_bins
    )

    # Модель tr2vec
    model = Transaction2VecJoint(
        amnt_loss,
        mcc_vocab_size,
        mcc_emb_size,
        amnt_bins,
        amnt_emb_size,
        emb_size,
        lr
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=5,
        verbose=False,
        mode='min'
    )

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        mode='min'
    )

    trainer = Trainer(
        accumulate_grad_batches=5,
        gpus=1,
        default_root_dir=logging_dir,
        deterministic=True,
        callbacks=[early_stop_callback, checkpoint],
        max_epochs=epochs,
        auto_lr_find=True
    )
    # Обучение
    trainer.fit(model, datamodule)
