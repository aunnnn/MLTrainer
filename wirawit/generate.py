import torch
import torch.nn.functional as F
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm_notebook as tqdm

sys.path.append('../')
from shared.models.basic_lstm import BasicLSTM
from shared.models.multi_layer_lstm import MultiLayerLSTM
from shared.process.pa4_dataloader import build_all_loaders
from shared.process.PA4Trainer import get_computing_device

import datetime
currentDT = datetime.datetime.now()
print("Start time:", currentDT.strftime("%Y-%m-%d %H:%M:%S"))

computing_device = get_computing_device()
all_loaders, infos = build_all_loaders('../pa4Data/')

char2ind = infos['char_2_index']
ind2char = infos['index_2_char']

# model = BasicLSTM(len(char2ind), 100, len(char2ind))
model = MultiLayerLSTM(len(char2ind), 120, num_layers=7, num_output=len(char2ind))
model.load_state_dict(torch.load('./multi7_lstm120/model_state.pt'))
model.eval()
model.to(computing_device)


prime_str = "<start>"
prime_tensor = torch.zeros(len(prime_str), len(char2ind)).to(computing_device)
        
for i in range(len(prime_str)):
    char = prime_str[i]
    prime_tensor[i, char2ind[char]] = 1    
    
    
# Sample from a category and starting letter
def sample(model, T=None, max_length = 2000, stop_on_end_tag=False):
    
    sample_music = ""
    
    with torch.no_grad():  # no need to track history in sampling
        model.reset_hidden(computing_device)
        
        # Prime with <start>, hidden state is now ready
        logits = model(prime_tensor.unsqueeze(dim=0))[-1]
        
        i = 0
        while i < max_length:
            res_ind = None
            if T is None:
                res_ind = np.argmax(logits).item()
            else:
                prob = np.array(F.softmax(logits/T, dim=0))
                res_ind = np.random.choice(len(char2ind), 1, p=prob)[0]
            final_char = ind2char[res_ind]            
            sample_music += final_char
            i+=1
            if i % 50 == 0:
                print(i)
                
            if stop_on_end_tag and (sample_music[-5:] == "<end>" or sample_music[-7:] == "<start>"):
                print("Found <end>, stop making music at i = {0}.".format(i))
                break
                
            next_char_tensor = torch.zeros(len(char2ind)).to(computing_device)
            next_char_tensor[res_ind] = 1
            next_char_tensor = next_char_tensor.view(1,1,-1)
            logits = model(next_char_tensor)[-1]
        return sample_music
    
m1 = sample(model, T=1, max_length=20_000, stop_on_end_tag=False)
m1 = m1.replace("<end>", "").replace("<start>", "")

with open("gen_samples.txt", "w") as tf:
    print(m1, file=tf)