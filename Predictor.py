import torch
from torch import nn

from Config import NUM_CLASSES

BOXES_PER_LOCATION = [6, 6, 6, 6, 4, 4]
INPUT_CHANNELS = [256, 12, 24, 48, 96, 192]
# INPUT_CHANNELS = [12, 24, 24, 48, 96, 192]



class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        for boxes_per_location, input_channels in zip(BOXES_PER_LOCATION, INPUT_CHANNELS):
            self.cls_headers.append(self.cls_block(input_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(input_channels,  boxes_per_location))
        self.reset_parameters()

    def cls_block(self, input_channels, boxes_per_location):
        return nn.Conv1d(input_channels, boxes_per_location * NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, input_channels, boxes_per_location):
        return nn.Conv1d(input_channels, boxes_per_location * 2, kernel_size=3, stride=1, padding=1)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            # feature = torch.DoubleTensor(feature).to('cuda')
            cls_logits.append(cls_header(feature))
            bbox_pred.append(reg_header(feature))

        batch_size = len(features[0])
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, NUM_CLASSES)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 2)
        return cls_logits, bbox_pred


if __name__ == '__main__':
    # 运行 查看生成的 检测框
    boxes = Predictor()
    boxes()
    print(len(boxes))
