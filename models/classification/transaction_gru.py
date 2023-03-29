from typing import Any, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.common_layers import PositionalEncoding
from models.classification import BaseModel
from utils.data_utils import global_context_emb_avg


class TransactionGRU(BaseModel):

    def __init__(
        self,
        emb_type:       str,
        mcc_vocab_size: int,
        mcc_emb_size:   int,
        amnt_bins:      int,
        amnt_emb_size:  int,
        emb_size:       int,
        hidden_size:    int,
        output_dim:     int,
        num_layers:     int,
        dropout:        float,
        lr:             float,
        gc_type:        int,
        is_gc:          bool,
        is_perm:        bool,
        is_pe:          bool,
        *args:          Any,
        **kwargs:       Any
    ) -> None:
        super().__init__(output_dim, *args, **kwargs)
        assert emb_type in ('concat', 'tr2vec')

        if emb_type == 'concat':
            emb_size = mcc_emb_size + amnt_emb_size
            if is_gc:
                if gc_type == 0:
                    emb_size += (amnt_emb_size + 3 * mcc_emb_size)
                elif gc_type == 1:
                    emb_size += amnt_emb_size + mcc_emb_size

        self.save_hyperparameters({
            'emb_type'          : emb_type,
            'mcc_vocab_size'    : mcc_vocab_size,
            'mcc_emb_size'      : mcc_emb_size,
            'amnt_bins'         : amnt_bins,
            'amnt_emb_size'     : amnt_emb_size,
            'emb_size'          : emb_size,
            'hidden_size'       : hidden_size,
            'output_dim'        : output_dim,
            'num_layers'        : num_layers,
            'dropout'           : dropout,
            'lr'                : lr,
            'is_gc'             : is_gc,
            'gc_type'           : gc_type,
            'is_perm'           : is_perm,
            'is_pe'             : is_pe
        })

        self.mcc_embeddings     = nn.Embedding(
            self.hparams['mcc_vocab_size'] + 1,
            self.hparams['mcc_emb_size'],
            padding_idx=0
        )
        self.amnt_embeddings    = nn.Embedding(
            self.hparams['amnt_bins'] + 1,
            self.hparams['amnt_emb_size'],
            padding_idx=0
        )

        if self.hparams['emb_type'] == 'concat':
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

        self.predictor = nn.Linear(2 * self.hparams['hidden_size'], self.hparams['output_dim'])

        self.gc_emb_amnt: Optional[torch.Tensor] = None
        self.gc_emb_mcc: Optional[torch.Tensor] = None

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


    def use_avg_emb_gc(self, dataset):
        self.gc_emb_amnt, self.gc_emb_mcc = global_context_emb_avg(
            dataset,
            self.amnt_embeddings,
            self.mcc_embeddings,
            device=self.device
        )


    def forward(
        self,
        mcc_seqs: torch.Tensor,
        amnt_seqs: torch.Tensor,
        lengths: torch.Tensor,
        avg_amnt: Optional[torch.Tensor],
        top_mcc: Optional[torch.Tensor],
        gc_ids: Optional[torch.Tensor]
    ) -> Any:
        mcc_embs = self.mcc_embeddings(mcc_seqs)
        amnt_embs = self.amnt_embeddings(amnt_seqs)
        embs = torch.cat([mcc_embs, amnt_embs], -1)
        if self.hparams['is_gc']:
            if self.hparams['gc_type'] == 0:
                avg_amnt_embs = self.amnt_embeddings(avg_amnt)
                mcc_top_embs = self.mcc_embeddings(top_mcc)
                mcc_top_embs = torch.reshape(mcc_top_embs, (mcc_top_embs.shape[0], mcc_top_embs.shape[1], -1))
                embs = torch.cat([embs, avg_amnt_embs, mcc_top_embs], -1)
            elif self.hparams['gc_type'] == 1:
                avg_amnt_embs = torch.zeros(amnt_embs, device=self.device, requires_grad=True)
                avg_mcc_embs = torch.zeros(mcc_embs, device=self.device, requires_grad=True)
                for x in gc_ids:
                    ...

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
