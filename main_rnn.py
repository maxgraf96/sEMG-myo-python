import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from DataModule import SEMGDataModule
from hyperparameters import BATCH_SIZE, NUM_EPOCHS
from sEMGRNN import SEMGRNN, model_name

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    data_module = SEMGDataModule(
        data_dir="data",
        batch_size=BATCH_SIZE
        )

    model = SEMGRNN()

    # If using wandb
    # wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        check_val_every_n_epoch=5,

        # CPU
        accelerator="cpu",

        # Single GPU
        # accelerator="gpu",
        # devices=1,

        # Or if you have multiple GPUs
        # devices=4,
        # strategy="ddp",

        # If using wandb
        # logger=wandb_logger
        )
    trainer.fit(model, data_module)

    torch.save(model.state_dict(), model_name + ".pt")
