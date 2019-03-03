#########################################################
# IMPORTANT: Assume this script is placed under some folder
# Because it will load `shared` lib from PARENT
#########################################################
import sys
import os
sys.path.append('../')
from shared.process.pa4_dataloader import build_all_loaders
from shared.models.basic_lstm import BasicLSTM
from shared.models.multi_layer_lstm import MultiLayerLSTM
from shared.process.PA4Trainer import PA4Trainer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


SESSION_NAME = "lstm100_300epochs"
PARENT_PATH_TO_SAVE_RESULT = './'

print("Session: " + SESSION_NAME)
print("---------------")

print("Loading data...")

LEARNING_RATE = 0.0001

all_loaders, other_infos = build_all_loaders('../pa4Data/', chunk_size=200, customize_loader_params={
    'num_workers': 2,
})

char_index = other_infos['char_2_index']

print("Done.")

train_loader = all_loaders['train']
val_loader = all_loaders['val']
test_loader = all_loaders['test']

INPUT_SIZE = len(char_index)
HIDDEN_SIZE = 100
# NUM_LAYERS = 7
OUTPUT_SIZE = len(char_index)

# model = MultiLayerLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
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

os.makedirs(os.path.join(PARENT_PATH_TO_SAVE_RESULT, SESSION_NAME), exist_ok=True)
log = open(logsavepath, "w")
sys.stdout = log
sys.stderr = log

import datetime
currentDT = datetime.datetime.now()
print("Start time:", currentDT.strftime("%Y-%m-%d %H:%M:%S"))

trainer = PA4Trainer(model, criterion, optimizer, all_loaders, {
    'path_to_save_result': PARENT_PATH_TO_SAVE_RESULT,
    'session_name': SESSION_NAME,
    'n_epochs': 300,
    'print_every_n_epochs': 5,
    'validate_every_v_epochs': 10,
    'verbose': True,
    'num_epochs_no_improvement_early_stop': 5,
    'use_early_stop': False,
    'pass_hidden_states_between_epochs': False,
})
trainer.start()