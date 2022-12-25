import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from week13.Config import device


class TPN_model(nn.Module):

    def __init__(self,INPUT_SENSOR_CHANNELS=3,OUTPUT_SENSOR_CHANNELS=150):
        super(TPN_model, self).__init__()
        self.conv1 = nn.Conv1d(INPUT_SENSOR_CHANNELS, 32, kernel_size=24, stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=16, stride=1)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=8, stride=1)
        self.maxPool1d = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.1)
        self.liner = nn.Linear(1, OUTPUT_SENSOR_CHANNELS)
        # self.init_weights()


    def forward(self, input):
        features = []
        x = F.relu(self.conv1(input))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.maxPool1d(x)
        # x = x.permute(0,2,1)
        x = F.relu(self.liner(x))
        x = x.permute(0, 2, 1)
        features.append(x.type(torch.DoubleTensor).to(device))
        return features

    def init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            torch.nn.init.orthogonal_(m.weight)
            m.bias.data.fill_(0)


# tpn_model = TPN_model(3)
# input = Variable(torch.randn(2, 3, 300))
# tpn_model(input)