import torch
import torch.nn as nn
import torch.nn.init as init

class VanillaRNN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, computing_device=None):
        super(VanillaRNN, self).__init__()
        
        self.computing_device = computing_device
        
        self.num_output = num_output
        self.num_hidden = num_hidden
        
        self.i2h = nn.Linear(num_input+num_hidden, num_hidden)
        self.h2o = nn.Linear(num_hidden, num_output)
        self.tanh = nn.Tanh()
        
        self.hidden = None
       
        init.xavier_normal_(self.i2h.weight)
        init.xavier_normal_(self.h2o.weight)
        
    
    def forward(self, input): 
        if self.hidden is None:
            self.hidden = self.init_hidden(self.computing_device)
            print("init new hidden")
         
        input_combined = torch.cat((input, self.hidden), 1)
        
        # Input+Hidden to Hidden
        a = self.i2h(input_combined)
        
        # Activation
        self.hidden = self.tanh(a)
        
        # Hidden to Output
        output = self.h2o(self.hidden)
        
        return output
   
    def reset_hidden(self, computing_device=None):
        self.hidden = None
   
    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = (self.hidden.detach())
    
    # Helper to init hidden state
    def init_hidden(self, computing_device=None):
        h = torch.zeros(1, self.num_hidden)
        if computing_device:
            return h.to(computing_device)
        else:
            return h
            