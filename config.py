# Configuration for training
training_config = {
    "batch_size": 40,
    "learning_rate": 5.e-4,
    "num_steps": 80000,
    "log_interval": 100,
    "decay_flag": False
}


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 32
    dropout: float = 0.05
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pos_flag: bool = True
    rec_flag: bool = False #True: recurrent
    mem_flag: bool = False #True: accumulate oldest state, else forget (for recurrent only)
    w0: int = 32
    w1: int = 32




