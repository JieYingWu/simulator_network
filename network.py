import torch
from torch import nn

# Testing fully connected networks for predicting positions
class SpringNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(SpringNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        x = self.activation(x)
        x = self.l4(x)
        
        return x

    
    
