import torch
import torch.nn as nn
import torch.nn.init as init

class BasicLSTM(nn.Module):
    """
    Since forward's outputs are logits, use with criterion nn.CrossEntropyLoss().
    NOTE: Work with only one batch input.
    """
    
    def __init__(self, num_input, num_hidden, num_output):
        super(BasicLSTM, self).__init__()
        
        self.num_hidden = num_hidden
        
        self.lstm = nn.LSTM(num_input, num_hidden, batch_first=True)
        self.h2o = nn.Linear(num_hidden, num_output)       

        init.xavier_normal_(self.h2o.weight)
    
    def forward(self, input):        
        """
        Input with shape [1, chunk_size, num_feature]
        E.g., [1, chunk_size, 92]
        """
        batch_size, chunk_size, feature_size = input.size()
        
        # input + hidden to hidden
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        
        # (chunk_size, num_hidden)
        linear_input = lstm_out.contiguous().view(-1, self.num_hidden)
        logits = self.h2o(linear_input)
        return logits
    
    def detach_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
    def reset_hidden(self, computing_device):
        self.hidden = self.init_hidden(computing_device)
        
    # Helper to init hidden state
    def init_hidden(self, computing_device=None):
        # (batch_size, num_layers, num_hidden)
        if computing_device:
            return (torch.zeros(1, 1, self.num_hidden).to(computing_device), 
                    torch.zeros(1, 1, self.num_hidden).to(computing_device))
        else:
            return (torch.zeros(1, 1, self.num_hidden), torch.zeros(1, 1, self.num_hidden))