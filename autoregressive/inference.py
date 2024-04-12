import os
import torch
from pathlib import Path
from miditok import REMIPlus
from lightning_model import LightningMusicTransformer

input_midi_folder=""

# load saved tokeniser 
tokeniser = REMIPlus(params=Path("/homes/erv01/Overpainting/preprocessing/REMIPlus_PiJAMA_data",
                                  "tokenizer_params.json")) # change the path?
print(len(tokeniser))

# load the saved model
# model = LightningMusicTransformer.load_from_check_point("/path/to/checkpoint.ckpt)

# load input seed/primer that you want to use 
# this will probably need to be a function that goes through a list of unseen MIDI data
# this also needs to be passed through the tokeniser before it can be used to generate

def get_midi_and_tokenise(input_midi_path, tokeniser_params):
    pass
# for loop walking through input_midi_folder
# take one MIDI
    



# Will also do it for seen MIDI data as well


# Code to generate the output from the model (need to adjust for my MIDI)
      # rand_seq1 = model.generate(torch.Tensor(out[-number_of_prime_tokens:]), target_seq_length=1024)
      # out1 = rand_seq1[0].cpu().numpy().tolist()
      # out_all.append(out1[number_of_prime_tokens:])
      # out = out1[number_of_prime_tokens:]


# Convert the generated tokens back to MIDI

if __name__ == '__main__':
    
    # 
    







# #Auto-Regressive Generator from miditok

# #NOTE: You much generate a seed composition first or it is not going to start

# number_of_cycles_to_run = 5 #@param {type:"slider", min:1, max:50, step:1}
# number_of_prime_tokens = 128 #@param {type:"slider", min:64, max:256, step:64}

# print('=' * 70)
# print('MidiTok Auto-Regressive Model Generator')
# print('=' * 70)
# print('Starting up...')
# print('=' * 70)
# print('Prime length:', len(out))
# print('Prime tokens:', number_of_prime_tokens)
# print('Prime input sequence', out[-8:])

# if len(out) != 0:
#   print('=' * 70)
#   out_all = []
#   out_all.append(out)
#   for i in tqdm(range(number_of_cycles_to_run)):
#       rand_seq1 = model.generate(torch.Tensor(out[-number_of_prime_tokens:]), target_seq_length=1024)
#       out1 = rand_seq1[0].cpu().numpy().tolist()
#       out_all.append(out1[number_of_prime_tokens:])
#       out = out1[number_of_prime_tokens:]
      
#       print(chr(10))
#       print('=' * 70)
#       print('Block number:', i+1)
#       print('Composition length so far:', (i+1) * 1024, 'notes')
#       print('=' * 70)

#   print('Done!' * 70)
#   print('Total blocks:', i+1)
#   print('Final omposition length:', (i+1) * 1024, 'notes')
#   print('=' * 70)

# OUT = []

# for o in out_all:
#     OUT.extend(o)
    
    
# converted_back_midi = tokenizer.tokens_to_midi([OUT], get_midi_programs(midi))
# converted_back_midi.dump('MidiTok-OUTPUT.mid')