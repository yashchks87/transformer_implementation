# Databricks notebook source
# !pip install wandb

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC import os
# MAGIC import sys
# MAGIC sys.path.append('/Workspace/Users/yash.choksi@intusurg.com/transformer_implementation/scripts/')
# MAGIC from prep_data import get_datasets
# MAGIC import torch
# MAGIC from model import Transformer
# MAGIC from tqdm import tqdm
# MAGIC import wandb
# MAGIC import torch
# MAGIC import torch.nn as nn
# MAGIC import torch.nn.functional as F
# MAGIC import pandas as pd

# COMMAND ----------

wandb.login(key='59b4b5d585fa30e485b9d49805d3f793de99f247')

# COMMAND ----------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multiple_gpus = True

batch_size = 92
train_loader, val_loader, test_loader = get_datasets('/Volumes/daai_ke_team/default/mfg_ai_images/translate_data/en-fr.csv', batch_size * 4, 0.01)

src_vocab_size = train_loader.dataset.english_tokenizer.vocab_size # type: ignore
tgt_vocab_size = train_loader.dataset.french_tokenizer.vocab_size # type: ignore
d_model = 512
# Number of heads in multi head attention
num_heads = 8
# num_layers here meaning how many stack of attention layers we are producing.
num_layers = 6
d_ff = 2048
max_seq_length = 35
dropout = 0.1

# loss_fn = nn.CrossEntropyLoss(ignore_index=0)

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# COMMAND ----------

def train_model(model, train_loader, val_loader, save_weights_path = None, load_model = None, wandb_dict = None, epochs = 10):
    if wandb_dict == None:
        wandb.init(project='english_to_french_translation')
    # else:
    #     wandb.init(project='video_classification', config=wandb_dict)
    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }
    if multiple_gpus == True:
        model = nn.DataParallel(model)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(epochs): 
        train_loss, val_loss = 0.0, 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            with tqdm(data_loaders[phase], unit="batch") as tepoch:
                for src, tgt in tepoch:
                    src, tgt = src.to(device), tgt.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        output, loss = model(src, tgt[:, :-1], tgt)
                        loss = loss.mean()
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss)
            if phase == 'train':
                train_loss = running_loss / len(train_loader)
            else:
                val_loss = running_loss / len(val_loader)
        wandb.log({
                    'train_loss': train_loss, 
                    'val_loss': val_loss,
                })
        print(f'Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}')

train_model(model, train_loader, val_loader, epochs = 50)

# COMMAND ----------



# COMMAND ----------


