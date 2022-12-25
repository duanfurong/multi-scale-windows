import torch
from torch import nn
import numpy as np
from torchstat import stat

from Config import NUM_CLASSES, seed
from LabelConfig import train_ground_target
from MultiBoxLoss import MultiBoxLoss
from Predictor import Predictor
from Priorbox import priorbox
from SelectConv import cnn
kernel_size = 3
dropout = 0.45

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(seed)


class ActivitySSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = Predictor().type(torch.DoubleTensor)
        self.priors = priorbox()().type(torch.DoubleTensor)
        self.loss = MultiBoxLoss(NUM_CLASSES, 0.5, 3)
        self.ground_target = train_ground_target
        # self.net = HARModel(NB_SENSOR_CHANNELS=NB_SENSOR_CHANNELS, n_hidden=OUT_SENSOR_CHANNELS, n_filters=OUT_SENSOR_CHANNELS, n_classes=NUM_CLASSES).type(torch.DoubleTensor).to(device)
        self.net = cnn().type(torch.DoubleTensor)
        # self.net = TPN_model(INPUT_SENSOR_CHANNELS=NB_SENSOR_CHANNELS).type(torch.DoubleTensor).to(device)


    def forward(self, input):
        self.net.train()
        features = self.net(input)
        # cls_logits, bbox_pred = self.predictor(features)
        return features

    def forward_with_postprocess(self, data,name):
        data = data.view(data.shape[0], 1, data.shape[1],data.shape[2])
        features = self.forward(data)
        cls_logits, bbox_pred = self.predictor(features)
        priors = priorbox()().to('cuda:0')
        prediction = (bbox_pred, cls_logits, priors)
        ground_target = [self.ground_target[num] for num in name]
        loss_l, loss_c,class_accuracy,back_accuracy,first_accuracy,second_accuracy,third_accuracy,fourth_accuracy = self.loss(prediction, ground_target)
        return loss_l, loss_c, class_accuracy,back_accuracy,first_accuracy,second_accuracy,third_accuracy, fourth_accuracy

    def forward_with_testprocess(self, data):
        # features = self.forward(data.permute(0,2,1))
        data = data.view(data.shape[0],1, data.shape[1], data.shape[2])
        features = self.forward(data)
        cls_logits, bbox_pred = self.predictor(features)
        priors = priorbox()().to('cuda:0')
        prediction = (bbox_pred, cls_logits, priors)
        return prediction


# model = ActivitySSD()
# stat(model, (1,200,3))