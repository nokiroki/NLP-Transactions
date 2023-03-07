from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConf:

    data_dir: str
    logging_dir: str
    emb_dir:str


@dataclass
class ModelConf:

    task                : str
    model_type          : str
    experiment_name     : str
    emb_weights_name    : str
    train_embeddings    : bool


@dataclass
class LearningConf:

    batch_size      : int
    lr              : float
    epochs          : int
    num_workers     : int
    n_experiments   : int


@dataclass
class ParamsConf:

    emb_type            : str
    mcc_vocab_size      : str
    mcc_embed_size      : int
    amnt_bins           : int
    amnt_emb_size       : int
    emb_size            : int
    layers              : int
    hidden_dim          : int
    dropout             : float
    permutation         : bool
    pe                  : bool
    use_global_features : bool
    global_features_step: int
    m_last              : int
    m_period            : int
    period              : str
    output_dim          : int
    num_heads           : Optional[int] = None
