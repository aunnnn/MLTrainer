import torch

class LSTM(torch.nn.Module):
    def __init__(self,input_dim=93, hidden_dim=100, output_dim=93, num_layers=1,batch_size=1):
        super(LSTM, self).__init__()
        
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        self.batch_size=1
        
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        
        # Define the LSTM layer
        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = torch.nn.Linear(self.hidden_dim, self.output_dim)
        torch.nn.init.xavier_normal_(self.linear.weight)
    
    def clear_hidden(self):
        #  shape for h + c states: (num_layers * num_directions, batch, hidden_size)
        #  self.hidden = (h,c)
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        
    def forward(self, input,computing_device):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of input:  (seq_len, batch, input_size)
        
        seq_len = input.size()[0]
        
        self.hidden = self.hidden.to(computing_device)
        input = input.to(computing_device)
        
        lstm_out, hidden = self.lstm(input.view(seq_len, self.batch_size, -1),self.hidden)
        
        # TODO detach hidden state from previous calcs?? 
        self.hidden = (hidden[0].detach(), hidden[1].detach())
        
        #print(lstm_out.size())
        
        # output of linear layer
        y_pred = self.linear(lstm_out.view(seq_len, self.hidden_dim))
        #y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred
    