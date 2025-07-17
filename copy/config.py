# Configuration for training
training_config = {
    #"batch_size": 64,
    "batch_size": 64,
    "learning_rate": 5.e-4,
    #"num_steps": 400000
    "num_steps": 40000,
    "log_interval": 100,
    "decay_flag": True
}

# Configuration for dataset
dataset_config = {
    #"l_noise": 4096,  # number of padding tokens
    "l_noise": 4096,  # number of padding tokens
    "l_memorize": 16,  # number of tokens to memorize
    "n_tokens": 16,  # alphabet size
    "lag": False,
    #"variable": True,  # Randomly distribute memorization tokens throughout sequence instead of frontloading them
    "variable": False,  # front load  memorization tokens
    "variable_length": False,  # Randomize number of tokens to memorize
    "one_hot": False,
    "reverse": False,
    "static": False,
}

class GPTConfig:
    block_size: int = 2*dataset_config['l_memorize'] + dataset_config['l_noise']
    vocab_size: int = dataset_config['n_tokens']
    n_layer: int = 1
    n_head: int = 6
    n_embd: int = 96
    dropout: float = 0.05
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pos_flag: bool = True
    rec_flag: bool = True #True: recurrent
    mem_flag: bool = True #True: accumulate oldest state, else forget
    w: int = 32




