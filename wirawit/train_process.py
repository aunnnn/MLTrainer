#########################################################
# IMPORTANT: Assume this script is placed under some folder
# Because it will load `shared` lib from PARENT
#########################################################
import sys
import os
sys.path.append('../')
from shared.process.pa4_dataloader import build_all_loaders
from shared.models.basic_lstm import BasicLSTM
from shared.process.PA4Trainer import PA4Trainer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


SESSION_NAME = "session_train_100_hiddens_lower_lr"
PARENT_PATH_TO_SAVE_RESULT = './'

print("Session: " + SESSION_NAME)
print("---------------")

print("Loading data...")

LEARNING_RATE = 0.01

all_loaders, char_index = build_all_loaders('../pa4Data/', chunk_size=100, customize_loader_params={
    'num_workers': 4,
})

print("Done.")

train_loader = all_loaders['train']
val_loader = all_loaders['val']
test_loader = all_loaders['test']

INPUT_SIZE = len(char_index)
HIDDEN_SIZE = 100
OUTPUT_SIZE = len(char_index)

model = BasicLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

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
    'n_epochs': 200,
    'print_every_n_epochs': 5,
    'validate_every_v_epochs': 5,
    'verbose': True,
    'num_epochs_no_improvement_early_stop': 3,
    'use_early_stop': True
})
trainer.start()