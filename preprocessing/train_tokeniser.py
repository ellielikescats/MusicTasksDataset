    
from miditok import REMIPlus, TokSequence
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path

data_root = "/import/c4dm-datasets/PiJAMA/"  # PiJAMA dataset location
tokens_root = "tokens"  # Output folder for tokens
tokens_bpe_root = "tokens_bpe"  # Output folder for tokens after applying BPE
tokenizer_folder = "REMIPlus_PiJAMA_data"  # Output folder for tokenizer parameters aka trained vocabulary
tokenizer = REMIPlus()

if __name__ == "__main__":
    # Create the directories if they do not exist
    Path(tokens_root).mkdir(parents=True, exist_ok=True)
    Path(tokens_bpe_root).mkdir(parents=True, exist_ok=True)

    # Clear the foldersv
    for folder in [tokens_root, tokens_bpe_root]:
        for file in Path(folder).glob("**/*.json"):
            file.unlink()

    # Convert MIDI files to tokens saved as JSON files
    midi_paths = list(Path(data_root).glob("**/*.mid"))

    # data_augmentation_offsets = [2, 1, 1]  # Uncomment if you want to use data augmentation
    # tokenizer.tokenize_midi_dataset(midi_paths, Path(tokens_root), data_augment_offsets=data_augmentation_offsets)
    tokenizer.tokenize_midi_dataset(midi_paths, Path(tokens_root))

    vocab_size = len(tokenizer.vocab)
    print(vocab_size)

    # Construct the vocabulary with BPE, from the tokenized files
    tokenizer.learn_bpe(
        vocab_size=2000,
        tokens_paths=list(Path(tokens_root).glob("**/*.json")),
        start_from_empty_voc=False,

    )

    vocab_size_bpe = len(tokenizer.vocab_bpe)
    print(vocab_size_bpe)
    # Save our tokenizer, to retrieve it back later with the load_params method
    tokenizer.save_params(Path(tokenizer_folder + "/tokenizer_params.json"))

