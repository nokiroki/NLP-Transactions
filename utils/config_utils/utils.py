from typing import Tuple
from configparser import ConfigParser
import os

from .config_data import ModelConf, ClassificationParamsConf, LearningConf, DataConf


def get_config_with_dirs(config_file: str) -> Tuple[
    Tuple[DataConf, ModelConf, LearningConf, ClassificationParamsConf],
    ConfigParser
]:
    config = ConfigParser()
    config.read(config_file, encoding='utf-8')

    data_conf = DataConf(
        config.get('Data', 'data_dir'),
        config.get('Logging', 'logging_dir'),
        config.get('Logging', 'emb_dir'),
        config.get('Data', 'dataset')
    )

    if not os.path.exists(data_conf.logging_dir):
        os.mkdir(data_conf.logging_dir)

    if not os.path.exists(data_conf.emb_dir):
        os.mkdir(data_conf.emb_dir)

    model_conf = ModelConf(
        config.get('All_models', 'task').lower(),
        config.get('All_models', 'model_type').lower(),
        config.get('All_models', 'experiment_name'),
        config.get('All_models', 'emb_weigths_name'),
        config.get('All_models', 'device')
    )

    conf_section = 'RNN' if model_conf.model_type == 'rnn' else 'Transformer'

    learning_conf = LearningConf(
        config.getint(conf_section, 'batch_size'),
        config.getfloat(conf_section, 'lr'),
        config.getint(conf_section, 'epochs'),
        config.getint('All_models', 'num_workers'),
        config.getint('All_models', 'n_experiments')
    )

    params_conf = ClassificationParamsConf(
        config.getboolean(conf_section, 'pretrained_embed'),
        config.getboolean(conf_section, 'train_embed'),
        config.get(conf_section, 'emb_type'),
        config.getint(conf_section, 'mcc_vocab_size'),
        config.getint(conf_section, 'mcc_emb_size'),
        config.getint(conf_section, 'amnt_bins'),
        config.getint(conf_section, 'amnt_emb_size'),
        config.getint(conf_section, 'emb_size'),
        config.getint(conf_section, 'layers'),
        config.getint(conf_section, 'hidden_dim'),
        config.getfloat(conf_section, 'dropout'),
        config.getboolean(conf_section, 'permutation'),
        config.getboolean(conf_section, 'pe'),
        config.getint('All_models', 'gc_type'),
        config.getboolean('All_models', 'use_global_features'),
        config.getboolean('All_models', 'is_weekends'),
        config.getint('All_models', 'global_features_step'),
        config.getint('All_models', 'm_last'),
        config.getint('All_models', 'm_period'),
        config.get('All_models', 'period'),
        config.getint(conf_section, 'output_dim'),
        config.getint(conf_section, 'n_heads') if model_conf.model_type == 'transformer' else None
    )

    return (data_conf, model_conf, learning_conf, params_conf), config


# Temp method to save logic
def get_config_with_dirs_old(config_file: str) -> Tuple[ConfigParser, Tuple[str, str]]:
    config = ConfigParser()
    config.read(config_file, encoding='utf-8')

    data_dir = config.get('Data', 'data_dir')
    logging_dir = config.get('Logging', 'logging_dir')
    emb_dir = config.get('Logging', 'emb_dir')
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)

    return config, (data_dir, logging_dir, emb_dir)
