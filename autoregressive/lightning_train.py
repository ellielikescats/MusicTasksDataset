import torch
import wandb
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, random_split
from lightning_datamodule import MIDIDataset, LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from lightning_model import LightningMusicTransformer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer


if __name__ == '__main__':

# Intialise wandb
    #wandb_logger = WandbLogger(log_model="all")
    pl.seed_everything(1)

    # # Define the device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # #device = torch.device('cpu')
    # # Path to save the model checkpoints
    # model_save_path = 'model_checkpoints/model_epoch{}.pth'
    # Paths to the directories containing the tokenized data
    x_directory = '/homes/erv01/Overpainting/preprocessing/tokens_x_labels'
    y_directory = '/homes/erv01/Overpainting/preprocessing/tokens_y_labels'
    csv_path = '/homes/erv01/Overpainting/preprocessing/tokens.csv'

    data_module = LightningDataModule(csv_path,x_directory,y_directory)


    # Define the model
    model = LightningMusicTransformer()
    trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp", 
                         #logger=WandbLogger, 
                         max_epochs=20)
    trainer.fit(model, data_module)
