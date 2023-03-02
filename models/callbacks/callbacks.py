from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule


class FreezeEmbeddings(Callback):

    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.mcc_embeddings.requires_grad_(False)
        pl_module.amnt_embeddings.requires_grad_(False)
        pl_module.emb_linear.requires_grad_(False)


class UnfreezeEmbeddings(Callback):
    
    def __init__(self, unfreeze_after_epoch=3) -> None:
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self.n_epoch = 0

    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.n_epoch == self.unfreeze_after_epoch:
            pl_module.embeddings.requires_grad_(True)
        self.n_epoch += 1
