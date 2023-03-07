from typing import Tuple
from configparser import ConfigParser
import os


def get_config_with_dirs(config_file: str) -> Tuple[ConfigParser, Tuple[str, str]]:
    config = ConfigParser()
    config.read(config_file, encoding='utf-8')

    data_dir = config.get('Data', 'data_dir')
    logging_dir = config.get('Logging', 'logging_dir')
    emb_dir = config.get('Logging', 'emb_dir')
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)

    return config, (data_dir, logging_dir, emb_dir)