import torch
import wandb
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, random_split
from DataModule import MIDIDataset
from torch.nn.utils.rnn import pad_sequence
from model2 import MusicTransformer



# Function for custom loss fuction used by Music Transformer
def loss_fn(prediction, target, criterion=nn.CrossEntropyLoss(reduction='none')):
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

# Custom function to collate the DataLoader and pad - this I've added myself, don't know if it works
# This is also batch padding
def pad_collate(batch):
    # Check if sequences are 1-dimensional and adjust accordingly
    # Assumes sequences are the first element if batch items are tuples/lists
    sequences = [item if isinstance(item, torch.Tensor) else item[0] for item in batch]
    
    for seq in sequences:
        print("Sequence Size for the first sequence is:",seq.size(0))
        break
    
    # Calculate the maximum sequence length
    max_len = max(seq.size(0) for seq in sequences)  # Use size(0) for 1D sequences
    print("Max Sequence length is:", max_len)

    # I'm not sure about this code here - ideally, I want to initialise torch zeroes to act as my pads. Then I want to find the max length and for each sequence apply those
    # pads to make up the difference between the max length and the sequence length. Padding needs to be applied to the right, and the padded sequences should then take up the max size and the batch size.
    # # Assuming sequences are numeric (float or int)
    padded_seqs = torch.zeros(len(batch), max_len, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        length = seq.size(0)  # Use size(0) for 1D sequences
        padded_seqs[i, :length] = seq  # No need to squeeze since we're assuming 1D sequences
    
    print("Length of padded seq is:", padded_seqs.size(0))
    return padded_seqs

def prepare_data(dataset_class, split_ratios, **kwargs):
    # Load and process data
    full_dataset = dataset_class(**kwargs)
    
    # Calculate split sizes
    train_size = int(split_ratios['train'] * len(full_dataset))
    val_size = int(split_ratios['val'] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Split data
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

# Intialise wandb
wandb.init(project="music-transformer")
# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# Path to save the model checkpoints
model_save_path = 'model_checkpoints/model_epoch{}.pth'
# Paths to the directories containing the tokenized data
x_data_path = '/homes/erv01/Overpainting/preprocessing/tokens_x_labels'
y_data_path = '/homes/erv01/Overpainting/preprocessing/tokens_y_labels'
tokens_path = '/homes/erv01/Overpainting/preprocessing/tokens.csv'

# Create the dataset
dataset = MIDIDataset(csv_path=tokens_path,x_directory=x_data_path,y_directory=y_data_path)
# print length of dataset 
print(len(dataset))

# Get the first item from the dataset
first_item = dataset[0]
second_item = dataset[1]

# Print the shape of the first item
print(first_item.shape, second_item.shape)

# Create test/val/train split

train_dataset, val_dataset, test_dataset = prepare_data(MIDIDataset,{'train': 0.8, 'val': 0.10, 'test': 0.10},
    csv_path=tokens_path,
    x_directory=x_data_path,
    y_directory=y_data_path,
)
print(len(train_dataset))
print(len(test_dataset))
print(len(val_dataset))

# Define batch size
batch_size = 16

# Create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

# test dataloader happens later in inference
# TEST TRAIN_DATALOADER
# for batch in train_dataloader:
#     data = batch  # Assuming batch directly gives you the tensor you're interested in
#     data = data.to(device)  # Move data to the same device your model will run on

#     print(data.shape)
   
#     # Shift the data to create the target sequence

#     input_seq = data[:, :-1]  # Exclude the last item of each sequence
#     target_seq = data[:, 1:]  # Exclude the first item (shifted version)
    
#     break  # We only want to test with the first batch


# Define the model
model = MusicTransformer()
model = model.to(device)

# Define the Loss Function
criterion = nn.CrossEntropyLoss()
# custom loss: 

# Define the optimizer
# lr = 1.0 # For optimiser with scheduler
# optimiser = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98)) # For optimiser with scheduler
optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define the scheduler
# warmup_steps = 4000
# scheduler = optim.lr_scheduler.LambdaLR(
#     optimizer,
#     lambda x: transformer_lr_schedule(emsize, x, warmup_steps)
# )


# Training loop
epochs = 10 # number of epochs
ntokens = 2000

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(train_dataloader):
        data = batch.to(device)
        print("Data shape:", data.shape)

        # from the model: x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)

        input_seq = data[:, :-1]  # Exclude the last item of each sequence
        target_seq = data[:, 1:]  # Exclude the first item (shifted version)
        print("Batch size:", data.size(0))
        print("Input seq size",input_seq.size(0))
        print("Input seq shape",input_seq.shape)

             
        optimiser.zero_grad()
        output = model(input_seq)
        print("Output shape before view:", output.shape)
        print("Output seq size",output.size(0))

        #loss = criterion(output.view(-1))
        print("Target sequence shape (before reshape):", target_seq.shape)

        reshaped_output = output.reshape(-1, ntokens)
        print("Reshaped output shape:", reshaped_output.shape)  
        reshaped_target = target_seq.reshape(-1)
        print("Reshaped target shape:", reshaped_target.shape)

        #loss = loss_fn(output.view(-1, ntokens), target_seq.view(-1))
        loss = loss_fn(reshaped_output, reshaped_target) # For custom loss function used by Music Transformer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimiser.step()
        #scheduler.step() # For optimiser with scheduler

        total_loss += loss.item()
        wandb.log({"Training Loss": loss.item()})

    print(f'Epoch: {epoch}, Loss: {total_loss / len(train_dataloader)}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')