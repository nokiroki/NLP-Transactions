import os

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.classification import TransactionGRU, Transformer
from models.callbacks import FreezeEmbeddings, UnfreezeEmbeddings
from datamodules import TransactionRNNDataModule
from datamodules.preprocessing import data_preprocessing
from utils.config_utils import get_config_with_dirs


if __name__ == '__main__':
    (data_conf, model_conf, learning_conf, params_conf), _ = get_config_with_dirs('config.ini')

    # Необходим файл с токеном для логгирование на comet
    with open('api_token.txt') as f:
        api_token = f.read()

    train_sequences, val_sequences, test_sequences = data_preprocessing(
        data_conf,
        model_conf,
        params_conf
    )

    results = list()
    
    #TODO сделать через интерфейс подгрузку весов
    if params_conf.pretrained_embed:
        weights = torch.load(os.path.join(
            data_conf.emb_dir,
            'embedding_weights',
            model_conf.emb_weights_name
        ))
    
    # Цикл обучения для оценки uncertainty
    for i in range(learning_conf.n_experiments):
        model = TransactionGRU(
            params_conf.emb_type,
            params_conf.mcc_vocab_size,
            params_conf.mcc_embed_size,
            params_conf.amnt_bins,
            params_conf.amnt_emb_size,
            params_conf.emb_size,
            params_conf.hidden_dim,
            params_conf.output_dim,
            params_conf.layers,
            params_conf.dropout,
            learning_conf.lr,
            params_conf.use_global_features,
            params_conf.permutation,
            params_conf.pe
        ) if model_conf.model_type == 'rnn' else Transformer(
            params_conf.emb_type,
            params_conf.mcc_vocab_size,
            params_conf.mcc_embed_size,
            params_conf.amnt_bins,
            params_conf.amnt_emb_size,
            params_conf.emb_size,
            params_conf.hidden_dim,
            params_conf.output_dim,
            params_conf.layers,
            params_conf.dropout,
            learning_conf.lr,
            params_conf.num_heads,
            params_conf.use_global_features,
            params_conf.permutation,
            params_conf.pe  
        )

        model.set_embeddings(weights['mccs'], weights['amnts'])

        datamodule = TransactionRNNDataModule(
            train_sequences,
            val_sequences,
            test_sequences,
            params_conf,
            learning_conf
        )

        early_stop_callback = EarlyStopping(
            monitor='val_auroc' if model_conf.task == 'classification' else 'val_accuracy',
            min_delta=1e-3,
            patience=4,
            verbose=False,
            mode='max'
        )

        checkpoint = ModelCheckpoint(
            monitor='val_auroc' if model_conf.task == 'classification' else 'val_accuracy',
            mode='max',
            dirpath=os.path.join(data_conf.logging_dir, 'checkpoints', model_conf.experiment_name))
        
        if params_conf.train_embed:
            callbacks = [checkpoint, early_stop_callback, UnfreezeEmbeddings()]
        else:
            callbacks = [checkpoint, early_stop_callback, FreezeEmbeddings()]

        comet_logger = CometLogger(
            api_key=api_token,
            project_name='NLP-transactions-next-token',
            experiment_name=f'{model_conf.experiment_name}_{i}'
        )

        trainer = Trainer(
            accelerator='gpu',
            devices=1,
            log_every_n_steps=20,
            logger=comet_logger,
            deterministic=True,
            callbacks=callbacks,
            max_epochs=learning_conf.epochs,
            auto_lr_find=True
        )

        trainer.fit(model, datamodule=datamodule)
        model_best = TransactionGRU.load_from_checkpoint(checkpoint.best_model_path) \
                    if model_conf.model_type == 'rnn' else \
                    Transformer.load_from_checkpoint(checkpoint.best_model_path)
        trainer.test(model_best, datamodule)
        # res = trainer.test(model_best, datamodule)[0]['test_auroc']
        # results.append(res)
