import torch
from torch import nn

class EVALUATE_MLP(nn.Module):
    def __init__(self, in_Channel, hidden_channels_list, out_Channel):
        super(EVALUATE_MLP, self).__init__()
        self.in_Channel = in_Channel
        self.hidden_channels_list = hidden_channels_list
        self.out_Channel = out_Channel
        self.mlp = nn.Sequential(nn.Linear(in_Channel, hidden_channels_list[0]),
                                nn.ReLU(True))
        for i in range(1, len(hidden_channels_list)):
            layer = nn.Linear(hidden_channels_list[i-1], hidden_channels_list[i])
            self.mlp = nn.Sequential(self.mlp, layer, nn.ReLU(True))
        # self.mlp = nn.Sequential(self.mlp, 
        #                     nn.Linear(hidden_channels_list[-1], out_Channel),
        #                     nn.Sigmoid()) 
        self.mlp = nn.Sequential(self.mlp, 
                            nn.Linear(hidden_channels_list[-1], out_Channel)) 
        
    def forward(self, x):
        out  = self.mlp(x)
        return out
        
    