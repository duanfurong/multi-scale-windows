from math import sqrt
import torch
from Boxs_op import center_form_to_corner_form, corner_form_to_center_form
from Config import ACTIVITY_SIZE


class priorbox:
    # FEATURE_MAPS = [1002, 501, 250, 125, 62, 31]
    # FEATURE_MAPS = [1002,501]
    FEATURE_MAPS = [100]
    # FEATURE_MAPS = [30, 15, 7, 3, 1]
    # MIN_SIZES = [400, 741, 1082, 1422, 1764, 2104]
    # MAX_SIZES = [200, 400, 741, 1082, 1422, 1764]
    MIN_SIZES = [39, 47, 69, 91, 112, 134]
    MAX_SIZES = [13, 25, 47, 69, 91, 112]
    ASPECT_RATIOS = [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    CLIP = True

    def __init__(self):
        self.activity_size = ACTIVITY_SIZE          # 模型输入活动的大小
        self.feature_maps = self.FEATURE_MAPS      # 特征活动图大小 [38,19,10,5,3,1]
        self.min_sizes = self.MIN_SIZES            # 检测框大框 [60, 111, 162, 213, 264, 315]
        self.max_sizes = self.MAX_SIZES            # [30, 60, 111, 162, 213, 264]
        self.aspect_ratios = self.ASPECT_RATIOS    # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = self.CLIP                      # True ,检测框越界截断.  0<检测框尺寸<300

    def __call__(self):
        priors = []
        for k, feature_map_h in enumerate(self.feature_maps):
            for i in range(feature_map_h):
                cy = (i + 0.5) / feature_map_h
                size = self.min_sizes[k]
                h = size / self.activity_size
                priors.append([cy, h])

                size = self.min_sizes[k] * self.max_sizes[k]
                h = size / self.activity_size
                priors.append([cy, h])

                size = self.min_sizes[k]
                h = size / self.activity_size
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cy, h / ratio])
                    priors.append([cy,  h * ratio])

        priors = torch.tensor(priors)

        priors.clamp_(max=1, min=0)
        return priors


if __name__ == '__main__':
    # 运行 查看生成的 检测框
    boxes = priorbox()()
    print(boxes)
    print(len(boxes))
