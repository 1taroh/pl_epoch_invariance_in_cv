import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, generator_seed=42):
        super().__init__()
        self.batch_size = batch_size
        if generator_seed is not None:
            self.generator = torch.Generator().manual_seed(generator_seed)
        else:
            self.generator = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            x = torch.arange(1,6, dtype=torch.float)
            y = torch.arange(1,6, dtype=torch.float)
            self.train_dataset = TensorDataset(x,y)
            self.val_dataset = TensorDataset(x,y)

    def train_dataloader(self):
        # 毎回Dataloaderをインスタンス化しているので，generatorを明示的に与えないと，validation_step等を実行すると乱数が進んで異なる結果になる．
        # check_val_every_n_epoch を変更すると結果が変わるのはこのためである．
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False)


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1,1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        print(x.item(), y.item())
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        print("------------------")
        print("validation_step")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

class EpochEndCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        print("------------------")


def main(check_val_every_n_epoch, generator_seed=None):
    pl.seed_everything(42, workers=True)
    # seed を固定してもdataloaderの順番は固定されない．
    # dataloaderの順番を固定するには外部からgeneratorを与える必要がある．
    
    epochs = 4

    dm = MyDataModule(generator_seed=generator_seed)
    model = MyModel()

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        enable_progress_bar=False,
        callbacks=[EpochEndCallback()],
        check_val_every_n_epoch=check_val_every_n_epoch,
        )

    trainer.fit(model, dm)

if __name__ == "__main__":
    check_val_every_n_epoch = int(sys.argv[1])
    # generator_seed が None の場合，dataloader の出力が変わる 
    # generator_seed を固定すると，dataloader の出力が check_val_every_epoch によらず固定される
    # generator_seed = None
    generator_seed = 42
    main(check_val_every_n_epoch, generator_seed)
