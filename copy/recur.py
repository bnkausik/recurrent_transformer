import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
#from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from config import training_config, dataset_config, GPTConfig
from block_model import GPT
from data_generator import generate_dataset


# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("using mps##################")
    ctx = torch.autocast(device_type="mps",dtype=torch.float16)
else:
    ctx = torch.autocast(device_type="cpu",dtype=torch.float16)
print(f'Using device: {device}')

# Define model
gptconfig = GPTConfig()
model = GPT(gptconfig)
model.to(device)

lr_0 = training_config["learning_rate"]
num_steps = training_config["num_steps"]
log_interval = training_config["log_interval"]
batch_size = training_config["batch_size"]
optimizer = optim.Adam(model.parameters(), lr=lr_0)
max_noise = dataset_config['l_noise']
w_cur = gptconfig.w


# Training function
def train():
    global w_cur,lr_0

    loss_arr = np.zeros(log_interval)
    acc_arr = np.zeros(log_interval)

    if gptconfig.rec_flag: dataset_config['l_noise'] = w_cur

    for step in range(num_steps):

        w_cur = int(0.5*dataset_config['l_noise'])
        if w_cur > gptconfig.w: w_cur = gptconfig.w

        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.train()
        with ctx:
            outputs,loss = model(inputs,targets,w_cur)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_arr[step%log_interval] = loss.item()
        acc_arr[step%log_interval] = 100*(torch.sum(outputs == targets)/torch.numel(targets)).item()

        if step and ((not step % log_interval) or (step == num_steps-1)):

            mean_loss = loss_arr.mean()
            mean_acc = acc_arr.mean()

            sem_loss = np.std(loss_arr)/np.sqrt(log_interval)
            sem_acc = np.std(acc_arr)/np.sqrt(log_interval)

            cur_lr = optimizer.param_groups[0]['lr']

            print(f'{step+1} {dataset_config['l_noise']} {w_cur} Samples: {step*batch_size} train_loss: {mean_loss:.3f} sem_loss: {sem_loss:.3e} train_acc: {mean_acc:.2f} sem_acc: {sem_acc:.3e}  lr: {cur_lr:.2e}', end=' ')

            vloss,vacc =validate()

            if vloss < 0.3:
                if dataset_config['l_noise'] < max_noise:
                    dataset_config['l_noise'] *= 2
                    if dataset_config['l_noise'] > max_noise:
                        dataset_config['l_noise'] = max_noise
                    if training_config['decay_flag']:
                        lr_0 = lr_0 * (1 - 0.5*dataset_config['l_noise']/max_noise)
                        for p in optimizer.param_groups: p['lr'] = lr_0

            if dataset_config['l_noise'] == max_noise and training_config['decay_flag']:
                new_lr = lr_0*(1 - np.exp(-vloss))
                if new_lr < cur_lr:
                    for p in optimizer.param_groups: p['lr'] = new_lr




# Validation function
def validate():
    loss_arr = np.zeros(log_interval)
    acc_arr = np.zeros(log_interval)

    model.eval()
    for step in range(log_interval):
        inputs, targets = generate_dataset(dataset_config, training_config)
        inputs = inputs.to(device)
        targets = targets.to(device)

        with ctx:
            outputs,loss = model(inputs,targets,w_cur)

        loss_arr[step] = loss.item()
        acc_arr[step] = 100*(torch.sum(outputs == targets)/torch.numel(targets)).item()

    mean_loss = loss_arr.mean()
    mean_acc = acc_arr.mean()
    sem_loss = np.std(loss_arr)/np.sqrt(log_interval)
    sem_acc = np.std(acc_arr)/np.sqrt(log_interval)
    print(f'vLoss: {mean_loss:.3f} sem_vloss: {sem_loss:.3e} vacc: {mean_acc:.2f} sem_vacc: {sem_acc:.3e}')
    return(mean_loss, mean_acc)



if __name__ == '__main__':
    print(training_config)
    print(dataset_config)
    print(vars(GPTConfig))
    train()

