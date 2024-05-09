import os
import json
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pandas as pd
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


# For reproducible train/val split
gen = torch.Generator()
gen.manual_seed(0)

batch_size = 32
num_workers = 4

def pad_collate(batch):
    # Check if sequences are 1-dimensional and adjust accordingly
    # Assumes sequences are the first element if batch items are tuples/lists
    sequences = [item if isinstance(item, torch.Tensor) else item[0] for item in batch]
    
    original_lengths = [seq.size(0) for seq in sequences]
    max_len = max(original_lengths)
    print("Max Sequence length is:", max_len)


    # I'm not sure about this code here - ideally, I want to initialise torch zeroes to act as my pads. Then I want to find the max length and for each sequence apply those
    # pads to make up the difference between the max length and the sequence length. Padding needs to be applied to the right, and the padded sequences should then take up the max size and the batch size.
    # # Assuming sequences are numeric (float or int)
    padded_seqs = torch.zeros(len(batch), max_len, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        length = seq.size(0)  # Use size(0) for 1D sequences
        padded_seqs[i, :length] = seq  # No need to squeeze since we're assuming 1D sequences
 
    all_correctly_padded = all(seq.size(0) == max_len for seq in padded_seqs)
    # print(f"All sequences padded correctly to max length: {all_correctly_padded}, Max length in this batch: {max_len}")
    return padded_seqs


class MIDIDataset(Dataset):
    def __init__(self, csv_path, x_directory, y_directory):
        self.mapping = pd.read_csv(csv_path)
        self.x_dir = x_directory
        self.y_dir = y_directory
        assert len(self.x_dir) == len(self.y_dir), "X and Y datasets must contain the same number of pairs"

        self.data = []

        # Fetch the corresponding row in the CSV using idx
        for idx in range(len(self.mapping)):
            row = self.mapping.iloc[idx]
            
            # Construct file paths based on the x_label and y_label columns
            x_file_path = os.path.join(self.x_dir, row['x_labels'])
            y_file_path = os.path.join(self.y_dir, row['y_labels'])

            # Load x and y sequences
            x_sequence = self.load_data(x_file_path)
            y_sequence = self.load_data(y_file_path)

            # Concatenate x and y sequences
            concatenated_seq = x_sequence + y_sequence

            self.data.append(concatenated_seq)


    def load_data(self, file_path):
        # Load data from a single JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    # Returns the number of MIDI files in the dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.long) # depending on how training goes, I might need to change to torch float type.
        # print(data_tensor.size()) # in this case, it comes up as a 1d tensor e.g ([314])
        # print(data_tensor.dim()) # again dimension here is 1
        return data_tensor

# dataset = MIDIDataset(csv_path=tokens_path, x_directory=x_data_path, y_directory=y_data_path)
# example_seq = dataset[0]
# example_seq_2 = dataset[1]
# print(example_seq)
    
class LightningDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, x_directory, y_directory):
        super().__init__()
        self.csv_path = csv_path
        self.x_dir = x_directory
        self.y_dir = y_directory
        # self.batch_size = batch_size
        self.dataset = MIDIDataset(self.csv_path,self.x_dir,self.y_dir)
        self.train = None
        self.val = None
        self.batch_size = batch_size
        
    def setup(self, stage: str = 'fit'):

        split_ratio = 0.9

        # Calculate train/validation split
        train_size = int(split_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size

        self.train, self.val = random_split(self.dataset, [train_size, val_size]
        #                             # ,generator=gen
                                    )# Use generator=gen for reproducible train/val split

        # train_indices = [i for i in range(train_size)]
        # val_indices = [train_size + i for i in range(len(self.dataset) - train_size)]
        # self.train = torch.utils.data.Subset(self.dataset, train_indices)
        # self.val = torch.utils.data.Subset(self.dataset, val_indices)
    
        # !Need to create a test dataset split
        
        # 1. Do I want to normalise the training data?
        # 2. Add any extra transformations to the training data?
        # 3. and same for the test data split


    def train_dataloader(self):
        train_dataset = DataLoader(self.train, batch_size=batch_size, collate_fn=pad_collate,
                                   num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
        # Things to consider:
        # Drop last batch? Pin memory?
        return train_dataset

    def val_dataloader(self):
        val_dataset = DataLoader(self.val, batch_size=batch_size,collate_fn=pad_collate,
                                 num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
        return val_dataset

    
    def test_dataloader(self):
        # Need to sp
        test_dataset = DataLoader(self.train, batch_size=batch_size,)
        return test_dataset


if __name__ == '__main__':
    x_directory = '/homes/erv01/Overpainting/preprocessing/tokens_x_labels'
    y_directory = '/homes/erv01/Overpainting/preprocessing/tokens_y_labels'
    csv_path = '/homes/erv01/Overpainting/preprocessing/tokens.csv'

    dm = LightningDataModule(csv_path=csv_path, x_directory=x_directory, y_directory=y_directory)
    dm.setup(stage="fit")

    # # Get a sample from the dataset
    item = dm.train[0]

    # # Print the shape of the sample
    print(item.shape)
    
    train_loader = dm.train_dataloader()

    # Fetch the first batch from the DataLoader
    first_batch = next(iter(train_loader))

    print(first_batch.shape)  # print the shape of the first batch [batch size, length of tensor]

