import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset

class MIDIDataset(Dataset):
    def __init__(self, csv_path, x_directory, y_directory):
        self.mapping = pd.read_csv(csv_path)
        self.x_data = x_directory
        self.y_data = y_directory
        
        assert len(self.x_data) == len(self.y_data), "X and Y datasets must contain the same number of pairs"


    def load_data(self, file_path):
        # Load data from a single JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    # Returns the number of MIDI files in the dataset
    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        # Fetch the corresponding row in the CSV using idx
        row = self.mapping.iloc[idx]
        
        # Construct file paths based on the x_label and y_label columns
        x_file_path = os.path.join(self.x_data, row['x_labels'])
        y_file_path = os.path.join(self.y_data, row['y_labels'])

        # Load x and y sequences
        x_sequence = self.load_data(x_file_path)
        y_sequence = self.load_data(y_file_path)

        # Concatenate x and y sequences

        concatenated_seq = x_sequence + y_sequence

        # You might want to return tensors, depending on your model requirements
        # For example, you can convert lists to PyTorch tensors here if needed

        data_tensor = torch.tensor(concatenated_seq, dtype=torch.long)
        # print(data_tensor.size()) # in this case, it comes up as a 1d tensor e.g ([314])
        # print(data_tensor.dim()) # again dimension here is 1

        return data_tensor

if __name__ == '__main__':

    x_data_path = '/homes/erv01/Overpainting/preprocessing/tokens_x_labels'
    y_data_path = '/homes/erv01/Overpainting/preprocessing/tokens_y_labels'
    tokens_path = '/homes/erv01/Overpainting/preprocessing/tokens.csv'

    dataset = MIDIDataset(csv_path=tokens_path, x_directory=x_data_path, y_directory=y_data_path)
    example_seq = dataset[0]
    example_seq_2 = dataset[1]

    #print(example_seq)