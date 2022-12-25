import torch
from torch import nn
import torch.nn.functional as F

from week19.Config import device, NB_SENSOR_CHANNELS


class HARModel(nn.Module):
    def __init__(self,NB_SENSOR_CHANNELS=3, n_hidden=6, n_layers=1, n_filters=6,
                 n_classes=5, filter_size=5, drop_prob=0.5):
        super(HARModel, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.conv1 = nn.Conv1d(NB_SENSOR_CHANNELS, n_filters, filter_size,padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size,padding=2)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size,padding=2)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size,padding=2)

        self.lstm1 = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers)

        self.fc = nn.Linear(300, 150)

        self.dropout = nn.Dropout(drop_prob)

        #第二个尺度
        # self.conv5 = nn.Conv1d(6, 12, filter_size, padding=2)
        # self.conv6 = nn.Conv1d(12, 12, filter_size, padding=2)
        # self.conv7 = nn.Conv1d(12, 12, filter_size, padding=2)
        # self.conv8 = nn.Conv1d(12, 12, filter_size, padding=2)
        #
        # self.lstm3 = nn.LSTM(12, 12, n_layers)
        # self.lstm4 = nn.LSTM(12, 12, n_layers)
        #
        # self.fc1 = nn.Linear(1002, 501)
        #
        # self.dropout = nn.Dropout(drop_prob)
        self.init_weights()

    def forward(self, x, hidden, hidden1, batch_size):
        features = []
        # x = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(batch_size, -1, self.n_filters)
        x, hidden = self.lstm1(x.permute(1,0,2), hidden)
        x, hidden = self.lstm2(x, hidden)
        x = x.permute(1,0,2)
        # x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.fc(x.permute(0,2,1))

        # out = x.view(batch_size, -1, self.n_classes)[:, -1, :]
        out = x.permute(0, 2, 1)
        features.append(out.type(torch.DoubleTensor).to(device))

        return features


    def init_hidden(self,train_on_gpu,  batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

    def init_hidden1(self, train_on_gpu, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, 12).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, 12).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, 12).zero_(),
                      weight.new(self.n_layers, batch_size, 12).zero_())

        return hidden
    def init_weights(m):
        if type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif type(m) == nn.Conv1d or type(m) == nn.Linear:
            torch.nn.init.orthogonal_(m.weight)
            m.bias.data.fill_(0)

