import miditok
from miditok import REMIPlus, TokSequence
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
from copy import deepcopy

import json

# Paths
tokenizer_params_path = "/homes/erv01/Overpainting/preprocessing/REMIPlus_PiJAMA_data/tokenizer_params.json"  # Path to saved tokenizer parameters
data_root_x_labels = "/homes/erv01/Overpainting/preprocessing/JAZZVAR_DATASET_2024/x_labels" # Data root of new dataset x labels
data_root_y_labels = "/homes/erv01/Overpainting/preprocessing/JAZZVAR_DATASET_2024/y_labels" # Data root of new dataset y labels
tokens_output_root_x = "tokens_x_labels"  # Output folder for tokenised x labels
tokens_output_root_y = "tokens_y_labels"# Output folder for tokenised y labels

# Create separate IDs for BOS and EOS tokens. This will be applied to all of the tokenised sequences.
# BOS_TOKEN_ID = 2050
# EOS_TOKEN_ID = 2550


# Load the tokenizer with the learned parameters
tokenizer = REMIPlus(params=tokenizer_params_path)
vocab_size = len(tokenizer.vocab)
BOS_TOKEN_ID = tokenizer.vocab['BOS_None']
EOS_TOKEN_ID = tokenizer.vocab['EOS_None']
print(vocab_size)

# Ensure the output directories exists
Path(tokens_output_root_x).mkdir(parents=True, exist_ok=True)
Path(tokens_output_root_y).mkdir(parents=True, exist_ok=True)

global_max_token_id = 0

# Function to tokenize MIDI files, add custom BOS and EOS tokens, and save
def tokenize_and_save(midi_paths, tokenizer, output_root):
    global global_max_token_id  # Indicate that we're using the global variable

    for midi_path in midi_paths:
        # Load the MIDI file
        midi = MidiFile(midi_path)
        print(midi)
        
        # # Tokenize the MIDI file to a sequence of tokens
        tokens = tokenizer.midi_to_tokens(midi, apply_bpe=True)  # Tokenize the MIDI file
        print("Token IDs:", tokens.ids)
        print("Max Token ID:", max(tokens.ids))
        print("Vocabulary Size:", len(tokenizer.vocab))  
        print("BPE Vocabulary Size:", len(tokenizer.vocab_bpe)) 
        current_max_id = max(tokens.ids)
        global_max_token_id = max(global_max_token_id, current_max_id)
        print("Global Max Token ID across all files:", global_max_token_id)


        # Add custom EOS and BOS tokens
        tokens_with_bos_eos = [BOS_TOKEN_ID] + tokens.ids + [EOS_TOKEN_ID]
        print(len(tokens_with_bos_eos))
        
        # Save the BPE-applied tokens 
        output_path = Path(output_root) / (midi_path.stem + '_tokens_bpe.json')
        with output_path.open('w') as f:
            # Save the token IDs post-BPE application
            json.dump(tokens_with_bos_eos, f)

        print(f"Processed and saved BPE tokens for: {midi_path.stem}")

# # Tokenize the labels
midi_paths_x_labels = list(Path(data_root_x_labels).glob("**/*.mid"))
midi_paths_y_labels = list(Path(data_root_y_labels).glob("**/*.mid"))
tokenize_and_save(midi_paths_x_labels, tokenizer, tokens_output_root_x)
tokenize_and_save(midi_paths_y_labels, tokenizer, tokens_output_root_y)

print("Tokenization complete.")


#     # Split MIDI paths in train/valid/test sets
#     total_num_files = len(midi_paths)
#     num_files_valid = round(total_num_files * 0.2)
#     num_files_test = round(total_num_files * 0.1)
#     shuffle(midi_paths)
#     midi_paths_valid = midi_paths[:num_files_valid]
#     midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
#     midi_paths_train = midi_paths[num_files_valid + num_files_test:]

#     # Loads tokens and create data collator
#     kwargs_dataset = {"min_seq_len": 256, "max_seq_len": 1024, "tokenizer": tokenizer}
#     dataset_train = DatasetTok(midi_paths_train, **kwargs_dataset)
#     dataset_valid = DatasetTok(midi_paths_valid, **kwargs_dataset)
#     dataset_test = DatasetTok(midi_paths_test, **kwargs_dataset)
#     collator = DataCollator(
#         tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"]
#     )
    
# # Save the datasets
# save_dataset(dataset_train, 'dataset_train.pkl')
# save_dataset(dataset_valid, 'dataset_valid.pkl')
# save_dataset(dataset_test, 'dataset_test.pkl')