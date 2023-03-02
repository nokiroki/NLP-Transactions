from typing import Any, List
from argparse import Namespace
import copy

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from .positional_encoding import PositionalEncoding


class TransactionTransformer(pl.LightningModule):
    def __init__(self, hparams):
        super(TransactionTransformer, self).__init__()
        self.save_hyperparameters(hparams)
        if isinstance(hparams, Namespace):
            hparams = vars(hparams)
        assert hparams['emb_type'] in ['concat', 'tr2vec']
        self.is_perm = hparams['permutation']
        if hparams['emb_type'] == 'concat':
            assert hparams['mcc_emb_size'] + hparams['amnt_emb_size'] == hparams['emb_size']
        self.lr = hparams['lr']
        self.N = hparams['layers']

        self.mcc_embeddings = nn.Embedding(hparams['mcc_vocab_size'] + 1, 
                                            hparams['mcc_emb_size'], 
                                            padding_idx=0)
        self.amnt_embeddings = nn.Embedding(hparams['amnt_bins'] + 1, 
                                            hparams['amnt_emb_size'], 
                                            padding_idx=0)

        if hparams['emb_type'] == 'concat':
            self.emb_linear = nn.Identity()
        else:
            self.emb_linear = nn.Linear(hparams['mcc_emb_size'] + hparams['amnt_emb_size'],
                                        hparams['emb_size'],
                                        bias=False)
        self.pos_enc = PositionalEncoding(hparams['emb_size'])

        basic_module = nn.TransformerEncoderLayer(hparams['emb_size'], hparams['n_heads'])
        self.encoder_layers = nn.ModuleList([copy.deepcopy(basic_module) for i in range(hparams['layers'])])

        self.predictor = nn.Linear(hparams['emb_size'], 1)

    def set_embeddings(self, mcc_weights, amnt_weights, emb_linear_weights=None):
        with torch.no_grad():
            self.mcc_embeddings.weight.data  = mcc_weights
            self.amnt_embeddings.weight.data = amnt_weights
            if emb_linear_weights is not None:
                self.emb_linear.weight.data = emb_linear_weights
  
    def forward(self, mcc_seqs, amnt_seqs, lengths):
        mcc_embs = self.mcc_embeddings(mcc_seqs)
        amnt_embs = self.amnt_embeddings(amnt_seqs)
        embs = torch.cat([mcc_embs, amnt_embs], -1)
        embs = self.emb_linear(embs)
        embs = self.pos_enc(embs)
        
        if self.is_perm:
            perm = torch.randperm(embs.size(1))
            embs = embs[:, perm, :]
        
        x = embs
        for i in range(self.N):
            x = self.encoder_layers[i](x) #, mask)

        features = self._mean_pooling(x, lengths)

        logits = self.predictor(features).squeeze()
        return logits

    def _mean_pooling(self, outputs, lengths):
        max_length = outputs.size(1)
        mask = torch.vstack([torch.cat([torch.zeros(length), 
                                        torch.ones(max_length - length)]) for length in lengths])
        mask = mask.bool().to(outputs.device).unsqueeze(-1)
        outputs.masked_fill_(mask, 0)
        feature_vector = outputs.sum(1) / lengths.unsqueeze(-1)
        return feature_vector

    def auroc(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        return roc_auc_score(labels.detach().cpu().numpy(), probs.detach().cpu().numpy())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {'optimizer': optimizer}
    
    def training_step(self, batch, batch_idx):
        mcc_seqs, amnt_seqs, labels, lengths = batch
        logits = self(mcc_seqs, amnt_seqs, lengths)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log('train_loss', loss)
        return {'loss': loss, 'probs': torch.sigmoid(logits), 'labels': labels}

    def training_epoch_end(self, outputs):
        probs  = torch.cat([o['probs']  for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])
        self.log('train_auroc', self.auroc(probs, labels), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        mcc_seqs, amnt_seqs, labels, lengths = batch
        logits = self(mcc_seqs, amnt_seqs, lengths)
        probs = torch.sigmoid(logits)

        return probs, labels

    def validation_epoch_end(self, outputs):
        probs, labels = zip(*outputs)
        probs, labels = torch.cat(probs), torch.cat(labels)
        self.log('val_auroc', self.auroc(probs, labels), prog_bar=True)

    def test_step(self, batch, batch_idx):
        mcc_seqs, amnt_seqs, labels, lengths = batch
        logits = self(mcc_seqs, amnt_seqs, lengths)
        probs = torch.sigmoid(logits)
        return probs, labels

    def test_epoch_end(self, outputs):
        probs, labels = zip(*outputs)
        probs, labels = torch.cat(probs), torch.cat(labels)
        self.log('test_auroc', self.auroc(probs, labels))