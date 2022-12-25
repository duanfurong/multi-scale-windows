import torch
from torch import nn


class SKConv(nn.Module):
    def __init__(self, infeatures,features, M, G, r, stride, L):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = int(features / r)
        print(d)
        self.M = M
        self.features = features
        self.infeatures = infeatures
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(infeatures, features, kernel_size=(3,1), stride=stride, padding=(1 + i,1), dilation=(1 + i,1), groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=(1,1), stride=(1,1), bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=(1,1), stride=(1,1))
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]
        feats = []
        for conv in self.convs:
            feats.append(conv(x))
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V


class cnn(nn.Module):
    def __init__(self, M=3, G=32, r=32, stride=1, L=16):
        super(cnn, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (5, 1), stride=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2_sk = SKConv(64, 128, M=M, G=G, r=r, stride=stride, L=L)

        self.conv3_sk = SKConv(128, 256, M=M, G=G, r=r, stride=stride, L=L)
        # self.se= SELayer(128,16)

        self.fc = nn.Sequential(
            nn.Linear(462, 100)
        )

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.conv2_sk(x)
        x = self.conv3_sk(x)
        # x = self.se(x)
        x = x.view(x.size(0), 256, -1)
        # print(x.shape)
        x = self.fc(x)
        # x = x.cuda()
        features.append(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        # print(x.shape)
        return features

conv = SKConv()