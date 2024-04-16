import torch
import json
from pathlib import Path
from miditok import REMIPlus
from miditoolkit import MidiFile
from lightning_model import LightningMusicTransformer

# CONFIGURATION
MODEL_CHECKPOINT_PATH = "/homes/erv01/Overpainting/autoregressive/checkpoints/lightning_logs/version_3/checkpoints/epoch=841-step=6736.ckpt"
TOKENISER_PARAMS_PATH = Path("/homes/erv01/Overpainting/preprocessing/REMIPlus_PiJAMA_data",
                                  "tokenizer_params.json")
MAX_GENERATION_LENGTH = 512
BATCH_DIMENSION = 0  # Typically 0 for most models

json_file_path = '/homes/erv01/Overpainting/autoregressive/inference/x_1_tokens_bpe.json'


def generate_fixed_length(model, primer_tokens, length):
    device = model.device
    input_tensor = torch.tensor(primer_tokens, dtype=torch.long).unsqueeze(0).to(device)  # Entire sequence so far
    
    current_input = input_tensor
    generated_tokens = []

    with torch.no_grad():
        for _ in range(length):  # Generate exactly 'length' tokens
            output = model(current_input)
            next_token = next_token.unsqueeze(1)
            next_token = output[:, -1, :].argmax(-1, keepdim=True)
            current_input = torch.cat((current_input, next_token), dim=1)
            generated_tokens.append(next_token.item())

    return generated_tokens

# Generate sequence until a stopping condition is met
def generate_until_stop(model, primer_tokens, stop_token_id, max_length):
    # Move model to device
    device = model.device

    # Convert primer tokens to tensor
    input_tensor = torch.tensor([primer_tokens], dtype=torch.long)
    print('Input tensor before unsqueeze', input_tensor.shape)
    input_tensor = input_tensor.to(device)
    print('Input tensor shape', input_tensor.shape)

    # Entire sequence so far
    current_input = input_tensor
    generated_sequence = []

    with torch.no_grad():
        while True:
            output = model(current_input)
            # Assuming output shape is [batch_size, sequence_length, num_tokens (or features)]
            # You should get logits for the last time step of the last sequence element
            last_token_logits = output[:, -1, :]  # This selects the last timestep; shape should be [1, num_tokens]
            print("last token logits shape", last_token_logits.shape)
            next_token_id = last_token_logits.argmax(-1).item()  # This should now correctly be a scalar

            # Append the predicted token ID to the generated sequence
            generated_sequence.append(next_token_id)

            # Stop if the EOS token is generated or if the maximum length is reached
            if len(generated_sequence) >= max_length:
            # if next_token_id == stop_token_id or len(generated_sequence) >= max_length:
                break

            # Prepare the next input to feed the model by appending the new token
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            current_input = torch.cat([current_input, next_token_tensor], dim=1)

    return generated_sequence

def load_json_tokens(file_path):
    with open(file_path, 'r') as file:
        token_data = json.load(file)
    return token_data

# Main function to orchestrate the pipeline
def main():

    # Load tokens from JSON file
    test_primer_tokens = load_json_tokens(json_file_path)

    # Load the saved model
    model = LightningMusicTransformer.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
    model = model.to('cuda')  # Explicitly move model to GPU
    model.eval()

    # Load saved tokeniser 
    tokeniser = REMIPlus(params=TOKENISER_PARAMS_PATH) # change the path?
    vocab_size = len(tokeniser.vocab)
    BOS_TOKEN_ID = tokeniser.vocab['BOS_None']
    EOS_TOKEN_ID = tokeniser.vocab['EOS_None']

    # # Generate with defined length
    # fixed_length_output = generate_fixed_length(model, test_primer_tokens, 100)  # Generate 100 tokens
    # print("Fixed length generation:", fixed_length_output)

    # Generate until EOS or max length
    undefined_length_output = generate_until_stop(model, test_primer_tokens, EOS_TOKEN_ID, MAX_GENERATION_LENGTH)
    print("Undefined length generation:", undefined_length_output)

    converted_back_midi = tokeniser.tokens_to_midi(undefined_length_output)
    converted_back_midi.dump('TEST.mid')

    
if __name__ == "__main__":
    main()


    # def generate_outputs(model,
#                   input_primer_tokens,
#                   no_steerable_cycles=5, # where this is the number of cycles the autoreg model goes through
#                   use_sliding_window=False,
#                   window_size=128
#                   ):
    
#     out = input_primer_tokens.copy()
#     out_all = [out.copy()]  # To store all generated outputs for final output
    
#     print('=' * 70)
#     print('Generating Variation')
#     print('=' * 70)
#     print('Starting up...')
#     print('=' * 70)
#     print('Primer length:', len(out))
#     print('Using sliding window approach' if use_sliding_window else 'Using entire sequence approach')

#     for i in tqdm(range(no_steerable_cycles)):
#         # Determine the portion of 'out' to use as input for the model
#         if use_sliding_window:
#             # print('Prime input sequence:', out[-window_size:])
#             input_tensor = torch.Tensor(out[-window_size:]).unsqueeze(0)  # Last part of the output
#         else:
#             input_tensor = torch.Tensor(out).unsqueeze(0)  # Entire sequence so far

#         # Generate new sequence
#         generation = model.generate(input_tensor) #target_seq_length=1024)device=torch.device("cuda")
#         output = generation[0].cpu().numpy().tolist()

#         # Append new output
#         if use_sliding_window:
#             out_all.append(output[window_size:])  # Skip the window part already used
#             out = out + output[window_size:]  # Append skipping the window part
#         else:
#             out_all.append(output)  # Append entire generated sequence
#             out = output  # Use the new output as the next input

#         print(chr(10))
#         print('=' * 70)
#         print('Block number:', i+1)
#         print('Composition length so far:', len(out), 'notes')
#         print('=' * 70)

#     print('Done!' + '=' * 70)
#     print('Total blocks:', len(out_all))
#     print('Final composition length:', len(out), 'notes')
#     print('=' * 70)

#     # Combine all parts into a final output
#     OUT = []
#     for o in out_all:
#         OUT.extend(o)

#     return OUT