import torch
from torch.utils import data
import os
from dataloader_v2 import *
from trainer_v2 import *
from lstm import *
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sys


def get_computing_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_char_index():
    all_loaders, char_index, index_char = build_all_loaders('../pa4Data/', chunk_size=100, customize_loader_params={
    'num_workers': 4,
    })
    return char_index

# Sample from a category and starting letter
def sample(model, char_index, computing_device, start_letter='%'):
    index_char = {char_index[char]: char for char in char_index.keys()}
    sample_text = "" + start_letter
    with torch.no_grad():  # no need to track history in sampling
        output = torch.zeros(1, len(char_index))
        model.reset_hidden(computing_device)

        output[0, char_index[start_letter]] = 1

        for i in range(max_length):
            output = model(output)
            topv, topi = output.topk(1)
            topi = topi[1][0]
            sample_text += index_char[topi]
            output = torch.zeros(1, len(char_index))
            output[0, char_index[topi]] = 1
    return sample_text

SESSION_NAME = "session_train_100_hiddens_adam"
PATH_TO_SAVE_RESULT = './'
save_folder_path = os.path.join(PATH_TO_SAVE_RESULT, SESSION_NAME)

EARLY_STOP_SAVE_PATH = os.path.join(save_folder_path, 'model_state_min_val_so_far.pt')

# dict_filename = "char_index_dict.pickle"
# dict_savepath = os.path.join(PATH_TO_SAVE_RESULT, SESSION_NAME, dict_filename)

# char_index = pickle.load(dict_savepath)
char_index = get_char_index()

print("Session (Generation): " + SESSION_NAME)
print("---------------")

INPUT_SIZE = len(char_index)
HIDDEN_SIZE = 100
OUTPUT_SIZE = len(char_index)

model = BasicLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load(os.path.join()))
computing_device = get_computing_device()
model.to(computing_device)
print("Model Loaded Best Checkpoint.")

print("Sample One Song")
