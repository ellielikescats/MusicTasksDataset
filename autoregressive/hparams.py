# default hparams for the model
hparams = {
    "d_model": 64,
    "num_layers": 4,
    "num_heads": 8,
    "d_ff": 512,
    "max_rel_dist": 1024,
    "max_abs_position": 0,
    "vocab_size": 2000, # 500 is the default vocab size for REMIPlus
    "bias": True,
    "dropout": 0.1,
    "layernorm_eps": 1e-6
}

# hparams for TF model - significantly larger
# hparams_large = {
#     "d_model": 256,
#     "num_layers": 6,
#     "num_heads": 8,
#     "d_ff": 1024,
#     "max_rel_dist": 1024,
#     "max_abs_position": 0,
#     "vocab_size": vocab_size,
#     "bias": True,
#     "dropout": 0.1,
#     "layernorm_eps": 1e-6
# }