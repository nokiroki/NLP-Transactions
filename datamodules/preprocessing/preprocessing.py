from typing import Tuple

import pandas as pd

from utils.config_utils import DataConf, ModelConf, ClassificationParamsConf
from .rosbank_preprocessing import rb_preprocessing


def data_preprocessing(
        data_conf: DataConf,
        model_conf: ModelConf,
        params_conf: ClassificationParamsConf
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data_conf.dataset == 'rosbank':
        return rb_preprocessing(data_conf, model_conf, params_conf)
    elif data_conf.dataset == 'tinkoff':
        raise NotImplementedError('Dataset preprocessing not implemented yet!')
    elif data_conf.dataset == 'gender':
        raise NotImplementedError('Dataset preprocessing not implemented yet!')
    else:
        raise AttributeError('Invalid dataset name!')
