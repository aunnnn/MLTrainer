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
        
        self.hidden = self.reset_hidden()
       
        init.xavier_normal_(self.i2h.weight)
        init.xavier_normal_(self.h2o.weight)
        
    
    def forward(self, input):
        if self.hidden is None:
            self.hidden = self.init_hidden(input.size()[1], self.computing_device)
           
        if self.hidden.size()[1] > input.size()[1]:
            self.full_size_hidden = self.hidden
            self.hidden = self.hidden[:, :input.size()[1], :]
        if self.hidden.size()[1] < input.size()[1]:
            self.hidden = self.full_size_hidden
        
                
        input_combined = torch.cat((input, self.hidden), 2)
        
        # Input+Hidden to Hidden
        a = self.i2h(input_combined)
        
        # Activation
        self.hidden = self.tanh(a)
        
        # Hidden to Output
        output = self.h2o(self.hidden)
        
        return output[0]
   
    def reset_hidden(self, computing_device=None):
        self.hidden = None
   
    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = (self.hidden.detach())
    
    # Helper to init hidden state
    def init_hidden(self, num_row, computing_device=None):
        h = torch.zeros(1, num_row, self.num_hidden)
        if computing_device:
            return h.to(computing_device)
        else:
            return h
            