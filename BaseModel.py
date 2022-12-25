import torch
from torch import nn

from Config import seed
from tcn import TemporalConvNet


torch.manual_seed(seed)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels,final_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(final_channels, output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1)
        return o
