import torch
from torch.utils import data
import os
from dataloader_v2 import *
from trainer_v2 import *
from lstm import *
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle


SESSION_NAME = "session_train_100_hiddens_adam"
PARENT_PATH_TO_SAVE_RESULT = './'

print("Session: " + SESSION_NAME)
print("---------------")

print("Loading data...")

LEARNING_RATE = 0.001

all_loaders, char_index, index_char = build_all_loaders('../pa4Data/', chunk_size=100, customize_loader_params={
    'num_workers': 4,
})

print("Done.")

import sys
train_loader = all_loaders['train']
val_loader = all_loaders['val']
test_loader = all_loaders['test']

INPUT_SIZE = len(char_index)
HIDDEN_SIZE = 100
OUTPUT_SIZE = len(char_index)

model = BasicLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Remove old log
nohuppath = os.path.join(PARENT_PATH_TO_SAVE_RESULT, 'nohup.out')
if os.path.exists(nohuppath):
    os.remove(nohuppath)
    print("Removed old nohup.out")

# PRINT TO .log FILE INSTEAD
logfilename = "log_{0}.log".format(SESSION_NAME)
logsavepath = os.path.join(PARENT_PATH_TO_SAVE_RESULT, SESSION_NAME, logfilename)
print("Redirecting all prints to {0}.".format(logsavepath))

# SAVE char_index dictionary as pickle file
dict_filename = "char_index_dict.pickle"
dict_savepath = os.path.join(PARENT_PATH_TO_SAVE_RESULT, SESSION_NAME, dict_filename)

os.makedirs(os.path.join(PARENT_PATH_TO_SAVE_RESULT, SESSION_NAME), exist_ok=True)
log = open(logsavepath, "w")
sys.stdout = log
sys.stderr = log

# dump the pickle file in the session directory
pickle.dump(char_index, dict_savepath)
print("Dumping char_index dictionary to {}".format(dict_savepath))

import datetime
currentDT = datetime.datetime.now()
print("Start time:", currentDT.strftime("%Y-%m-%d %H:%M:%S"))

trainer = PA4Trainer_v2(model, criterion, optimizer, all_loaders, {
    'path_to_save_result': PARENT_PATH_TO_SAVE_RESULT,
    'session_name': SESSION_NAME,
    'n_epochs': 200,
    'print_every_n_epochs': 5,
    'validate_every_v_epochs': 5,
    'verbose': True,
    'num_epochs_no_improvement_early_stop': 3, 
    'use_early_stop': True
})
trainer.start()