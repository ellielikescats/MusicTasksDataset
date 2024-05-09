import torch
from math import sqrt
from torch import nn
from masking import create_mask
from hparams import hparams
from layers import DecoderLayer, abs_positional_encoding
import pytorch_lightning as pl
import torch.nn.functional as F

from lightning_datamodule import batch_size

"""
Implementation of Music Transformer model adapted from https://github.com/jason9693/MusicTransformer-pytorch/blob, using torch.nn.TransformerDecoder
based on Huang et. al, 2018, Vaswani et. al, 2017 (code from 2021 Aditya Gomatam and https://github.com/jason9693/MusicTransformer-pytorch/blob).
Model based on 
"""

class LightningMusicTransformer(pl.LightningModule):
    """
    Transformer Decoder with Relative Attention. Consists of:
        1. Input Embedding
        2. Absolute Positional Encoding
        3. Stack of N DecoderLayers
        4. Final Linear Layer
    """
    def __init__(self,
                 d_model=hparams["d_model"],
                 num_layers=hparams["num_layers"],
                 num_heads=hparams["num_heads"],
                 d_ff=hparams["d_ff"],
                 max_rel_dist=hparams["max_rel_dist"],
                 max_abs_position=hparams["max_abs_position"],
                 vocab_size=2000,
                 bias=hparams["bias"],
                 dropout=hparams["dropout"],
                 layernorm_eps=hparams["layernorm_eps"]):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            d_ff (int): intermediate dimension of FFN blocks
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings. Set to 0 to compute normal attention
            max_abs_position (int): maximum absolute position for which to create sinusoidal absolute
                                    positional encodings. Set to 0 to compute pure relative attention
                                    make it greater than the maximum sequence length in the dataset if nonzero
            bias (bool, optional): if set to False, all Linear layers in the MusicTransformer will not learn
                                   an additive bias. Default: True
            dropout (float in [0, 1], optional): dropout rate for training the model. Default: 0.1
            layernorm_eps (very small float, optional): epsilon for LayerNormalization. Default: 1e-6
        """
        super(LightningMusicTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist,
        self.max_position = max_abs_position
        self.vocab_size = vocab_size
        self.ntokens = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = abs_positional_encoding(max_abs_position, d_model
                                                           #,device=self.device
                                                           )
        self.input_dropout = nn.Dropout(dropout)

        self.decoder = nn.TransformerDecoder(
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps),
            num_layers=num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        )

        self.final = nn.Linear(d_model, vocab_size)
        #self.save_hyperparameters # option to save hyperparameters (be careful as could be quite large!)

    def forward(self, x, mask=None):
        """
        Forward pass through the Music Transformer. Embeds x according to Vaswani et. al, 2017, adds absolute
        positional encoding if present, performs dropout, passes through the stack of decoder layers, and
        projects into the vocabulary space. DOES NOT SOFTMAX OR SAMPLE OUTPUT; OUTPUTS LOGITS.

        Args:
            x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)
            mask (optional): mask for input batch indicating positions in x to mask with 1's. Default: None

        Returns:
            input batch after above steps of forward pass through MusicTransformer
        """
        # embed x according to Vaswani et. al, 2017
        x = self.input_embedding(x)
        # print(x.shape)
        x *= sqrt(self.d_model)

        # add absolute positional encoding if max_position > 0, and assuming max_position >> seq_len_x
        if self.max_position > 0:
            x += self.positional_encoding[:, :x.shape[-2], :]

        # input dropout
        x = self.input_dropout(x)

        # pass through decoder
        x = self.decoder(x, memory=None, tgt_mask=mask)

        # final projection to vocabulary space
        return self.final(x)
    
    def custom_cross_entropy_loss(self, prediction, target, criterion=nn.CrossEntropyLoss(reduction='none')):
# """
#     Since some positions of the input sequences are padded, we must calculate the loss by appropriately masking
#     padding values
    
    # Question: if we mask using zeros - then doesn't that go against the point of the pad which are zeros?
    # Need to ask ChatGPT

#     Args:
#         prediction: output of the model for some input
#         target: true value the model was supposed to predict
#         criterion: vanilla loss criterion

#     Returns:
#         masked loss between prediction and target
#     """
        mask = torch.ne(target, torch.zeros_like(target))           # ones where target is 0
        _loss = criterion(prediction, target)     # loss before masking

        print("_loss shape:", _loss.shape)
        print("mask shape:", mask.shape)

        # multiply mask to loss elementwise to zero out pad positions
        mask = mask.to(_loss.dtype)
        _loss *= mask

        # output is average over the number of value
        return torch.sum(_loss) / torch.sum(mask)
    
    def training_step(self, train_batch, batch_idx):
        #self.log("train/loss", loss)
        data = train_batch
        # print("Data shape:", data.shape)

        # from the model: x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)

        input = data[:, :-1]  # Exclude the last item of each sequence
        target = data[:, 1:]  # Exclude the first item (shifted version)
        output = self(input, mask=create_mask(input, n=input.dim() + 2))

        # print("Input tensor size:", input.shape)
        # print("Target tensor size:", target.shape)
        # print("Output tensor size:", output.shape)

        # Cross-entropy loss wants input to be (BS, NUM_CLASSES, DIMENSIONSxyz...)
        # reshaped_output = output.reshape(batch_size, self.ntokens, -1)
        # reshaped_target = target.reshape(batch_size, -1)
        reshaped_output = output.transpose(-1, -2)
        reshaped_target = target

        # print("Reshaped Input tensor size:", input.shape)
        # print("Reshaped Target tensor size:", target.shape)

        loss = self.custom_cross_entropy_loss(reshaped_output, reshaped_target)
        # print("Loss tensor size:", loss.shape)

        self.log('train_loss', loss)
        # print("Training loss:", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        data = val_batch
        print("Data shape:", data.shape)

        # from the model: x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)

        input = data[:, :-1]  # Exclude the last item of each sequence
        target = data[:, 1:]  # Exclude the first item (shifted version)
        output = self(input, mask=create_mask(input, n=max(input.dim() + 2, 2)))

        # Cross-entropy loss wants input to be (BS, NUM_CLASSES, DIMENSIONSxyz...)
        # reshaped_output = output.reshape(batch_size, self.ntokens, -1)
        # reshaped_target = target.reshape(batch_size, -1)
        reshaped_output = output.transpose(-1, -2)
        reshaped_target = target

        loss = self.custom_cross_entropy_loss(reshaped_output,reshaped_target)
        

        self.log('val_loss', loss)
        print("Validation Loss:", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def on_epoch_end(self):
        # Save the model every 10th epoch
        if (self.trainer.current_epoch + 1) % 10 == 0:
            self.trainer.save_checkpoint("best_model.ckpt")
            print(f'Model saved at epoch {self.current_epoch + 1}')
