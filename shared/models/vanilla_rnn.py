import torch
import torch.nn as nn
import torch.nn.init as init

class VanillaRNN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(VanillaRNN, self).__init__()
        
        self.num_hidden = num_hidden
        
        self.i2h = nn.Linear(num_input+num_hidden, num_hidden)
        self.h2o = nn.Linear(num_hidden, num_output)
        self.tanh = nn.Tanh()
        
        init.xavier_normal_(self.i2h.weight)
        init.xavier_normal_(self.h2o.weight)
    
    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        
        # Input+Hidden to Hidden
        a = self.i2h(input_combined)
        
        # Activation
        hidden = self.tanh(a)
        
        # Hidden to Output
        output = self.h2o(hidden)
        
        return output, hidden
    
    # Helper to init hidden state
    def init_hidden(self, computing_device=None):
        h = torch.zeros(1, self.num_hidden)
        if computing_device:
            return h.to(computing_device)
        else:
            return h
            