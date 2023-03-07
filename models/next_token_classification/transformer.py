from typing import Any, Optional

import torch
from torch import nn

from models.common_layers import PositionalEncoding
from models.next_token_classification import BaseModel


class Transformer(BaseModel):

    def __init__(
        self,
        emb_type: str,
        mcc_vocab_size: int,
        mcc_emb_size: int,
        amnt_bins: int,
        amnt_emb_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        lr: float,
        n_heads: int,
        is_perm: bool = False,
        is_pe: bool = False,
        output_dim: int = 344,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters({
            'emb_type'          : emb_type,
            'mcc_vocab_size'    : mcc_vocab_size,
            'mcc_emb_size'      : mcc_emb_size,
            'amnt_bins'         : amnt_bins,
            'amnt_emb_size'     : amnt_emb_size,
            'emb_size'          : emb_size,
            'hidden_size'       : hidden_size,
            'num_layers'        : num_layers,
            'dropout'           : dropout,
            'lr'                : lr,
            'n_heads'           : n_heads,
            'is_perm'           : is_perm,
            'is_pe'             : is_pe,
            'output_dim'        : output_dim
        })

        self.mcc_embeddings     = nn.Embedding(mcc_vocab_size + 1, mcc_emb_size, padding_idx=0)
        self.amnt_embeddings    = nn.Embedding(amnt_bins + 1, amnt_emb_size, padding_idx=0)

        if emb_type == 'concat':
            self.emb_linear = nn.Identity()
        else:
            self.emb_linear = nn.Linear(mcc_emb_size + amnt_emb_size, emb_size)

        self.pos_enc = PositionalEncoding(emb_size) if is_pe else nn.Identity()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_size, n_heads, dropout=dropout),
            num_layers
        )

        self.predictor = nn.Linear(emb_size, output_dim)


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

        hidden = self.encoder(embs)

        features = self._mean_pooling(hidden, lengths)

        logits = self.predictor(features).squeeze()
        return logits
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return {'optimizer': optimizer}
