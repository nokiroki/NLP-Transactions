from typing import Any, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.common_layers import PositionalEncoding
from models.churn_classification import BaseModel
from utils.config_utils import LearningConf, ClassificationParamsConf


class TransactionGRU(BaseModel):

    def __init__(
        self,
        learning_conf: LearningConf,
        params_conf: ClassificationParamsConf,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        assert params_conf.emb_type in ('concat', 'tr2vec')

        if params_conf.emb_type == 'concat':
            params_conf.emb_size = params_conf.mcc_embed_size + params_conf.amnt_emb_size
            if params_conf.use_global_features:
                params_conf.emb_size += (params_conf.amnt_emb_size + 3 * params_conf.mcc_embed_size)

        self.save_hyperparameters({
            'emb_type'          : params_conf.emb_type,
            'mcc_vocab_size'    : params_conf.mcc_vocab_size,
            'mcc_emb_size'      : params_conf.mcc_embed_size,
            'amnt_bins'         : params_conf.amnt_bins,
            'amnt_emb_size'     : params_conf.amnt_emb_size,
            'emb_size'          : params_conf.emb_size,
            'hidden_size'       : params_conf.hidden_dim,
            'num_layers'        : params_conf.layers,
            'dropout'           : params_conf.dropout,
            'lr'                : learning_conf.lr,
            'is_perm'           : params_conf.permutation,
            'is_pe'             : params_conf.pe
        })

        self.mcc_embeddings     = nn.Embedding(
            self.hparams['mcc_vocab_size'] + 1,
            self.hparams['mcc_emb_size'],
            padding_idx=0
        )
        self.amnt_embeddings    = nn.Embedding(
            self.hparams['amnt_bins'] + 1,
            self.hparams['amnt_emb_size'],
            padding_idx=0)

        if params_conf.emb_type == 'concat':
            self.emb_linear = nn.Identity()
        else:
            self.emb_linear = nn.Linear(
                self.hparams['mcc_embed_size'] + self.hparams['amnt_emb_size'],
                self.hparams['emb_size'],
                bias=False
            )

        self.pos_enc = PositionalEncoding(self.hparams['emb_size']) if self.hparams['is_pe'] else nn.Identity()

        self.rnn = nn.GRU(
            self.hparams['emb_size'],
            self.hparams['hidden_size'],
            self.hparams['num_layers'],
            bidirectional=True,
            batch_first=True,
            dropout=self.hparams['dropout']
        )

        self.predictor = nn.Linear(2 * self.hparams['hidden_size'], 1)


    def set_embeddings(
        self,
        mcc_weights: torch.Tensor,
        amnt_weights: torch.Tensor,
        emb_linear_weights: Optional[torch.Tensor] = None
    ):
        with torch.no_grad():
            self.mcc_embeddings.weight.data = mcc_weights
            self.amnt_embeddings.weight.data = amnt_weights
            if emb_linear_weights:
                self.emb_linear.weight.data = emb_linear_weights


    def forward(
        self,
        mcc_seqs: torch.Tensor,
        amnt_seqs: torch.Tensor,
        lengths: torch.Tensor,
        avg_amnt: Optional[torch.Tensor],
        top_mcc: Optional[torch.Tensor]
    ) -> Any:
        mcc_embs = self.mcc_embeddings(mcc_seqs)
        amnt_embs = self.amnt_embeddings(amnt_seqs)
        embs = torch.cat([mcc_embs, amnt_embs], -1)
        if avg_amnt is not None and top_mcc is not None:
            avg_amnt_embs = self.amnt_embeddings(avg_amnt)
            mcc_top_embs = self.mcc_embeddings(top_mcc)
            mcc_top_embs = torch.reshape(mcc_top_embs, (mcc_top_embs.shape[0], mcc_top_embs.shape[1], -1))
            embs = torch.cat([embs, avg_amnt_embs, mcc_top_embs], -1)
        embs = self.emb_linear(embs)
        embs = self.pos_enc(embs)
        
        if self.hparams['is_perm']:
            perm = torch.randperm(embs.size(1))
            embs = embs[:, perm, :]

        packed_embs = pack_padded_sequence(
            embs, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        hidden, _ = self.rnn(packed_embs)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        features = self._mean_pooling(hidden, lengths)

        logits = self.predictor(features).squeeze()
        return logits
    

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return {'optimizer': optimizer}
