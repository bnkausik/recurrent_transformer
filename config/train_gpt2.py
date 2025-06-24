# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

#model size
#n_layer=12
#n_head=12
#n_embd=768

########override tweaks for dense LLM
compile=False
eval_interval = 1000
eval_iters=100
batch_size = 100
gradient_accumulation_steps = 1
block_size = 64 # context length
learning_rate = 1.e-4
min_lr = learning_rate/10
max_iters=600000
lr_decay_iters=max_iters
dropout=0.0
device="mps"
always_save_checkpoint = True
eval_only = False
ser_flag = False
kernel_width = 2

n_layer=1
n_head=12
n_embd=768


