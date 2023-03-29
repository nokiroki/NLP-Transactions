from models.classification import TransactionGRU, Transformer
from models.callbacks import FreezeEmbeddings, UnfreezeEmbeddings
from datamodules import TransactionRNNDataModule
from datamodules.preprocessing import data_preprocessing
from utils.config_utils import get_config_with_dirs

if __name__ == '__main__':
    (data_conf, model_conf, learning_conf, params_conf), _ = get_config_with_dirs('config.ini')

    original_df, (train_sequences, val_sequences, test_sequences) = data_preprocessing(
        data_conf,
        model_conf,
        params_conf
    )