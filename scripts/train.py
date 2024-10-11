import os
import sys
sys.path.append('../')
from prep_data import get_datasets
import torch
from model import Transformer
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# print('Hello')
# csv = pd.read_csv('/Volumes/daai_ke_team/default/mfg_ai_images/translate_data/temp.csv')
# print(csv.head())
# csv = csv[:10000]
# csv.to_csv('/Volumes/daai_ke_team/default/mfg_ai_images/translate_data/temp.csv', index=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multiple_gpus = True

batch_size = 24
train_loader, val_loader, test_loader = get_datasets('/Volumes/daai_ke_team/default/mfg_ai_images/translate_data/en-fr.csv', batch_size * 4, 0.01)

src_vocab_size = train_loader.dataset.english_tokenizer.vocab_size # type: ignore
tgt_vocab_size = train_loader.dataset.french_tokenizer.vocab_size # type: ignore
d_model = 512
# Number of heads in multi head attention
num_heads = 8
# num_layers here meaning how many stack of attention layers we are producing.
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)




def train_model(model, train_loader, val_loader, save_weights_path = None, load_model = None, wandb_dict = None, epochs = 10):
    if wandb_dict == None:
        wandb.init(project='english_to_french_translation')
    # else:
    #     wandb.init(project='video_classification', config=wandb_dict)
    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }
    # if load_model != None:
    #     model = model.load_state_dict(torch.load(load_model), strict=False)
    # if load_model != None:
    #     model.load_state_dict(torch.load(load_model)['model_state_dict'], strict=False)
    
    if multiple_gpus == True:
        model = nn.DataParallel(model)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
                        output = model(src, tgt[:, :-1])
                        output = output.permute(0, 2, 1)
                        tgt = tgt[:, 1:]
                        loss = loss_fn(output, tgt)
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
        #             'train_prec': train_prec,
        #             'val_prec': val_prec,
        #             'train_rec': train_rec,
        #             'val_rec': val_rec
                })
        print(f'Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}')
        # save_model(model, epoch, optimizer, False, save_weights_path)

train_model(model, train_loader, val_loader, epochs = 50)