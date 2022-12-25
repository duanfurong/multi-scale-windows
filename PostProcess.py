import torch
import torch.nn.functional as F
from Boxs_op import convert_locations_to_boxes, center_form_to_corner_form

from Config import ACTIVITY_SIZE, CENTER_VARIANCE, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, MAX_PER_IMAGE

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class postprocessor:
    def __init__(self):
        super().__init__()
        self.height = ACTIVITY_SIZE

    def __call__(self, cls_logits, bbox_pred, priors):
        batches_scores = F.softmax(cls_logits, dim=2)
        boxes = convert_locations_to_boxes(
            bbox_pred, priors, CENTER_VARIANCE, CENTER_VARIANCE
        )
        batches_boxes = center_form_to_corner_form(boxes)
        batches_boxes = batches_boxes.clamp(min=0, max=1)
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            processed_boxes = []
            processed_scores = []
            processed_labels = []

            per_img_scores, per_img_boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            for class_id in range(1, per_img_scores.size(1)):
                scores = per_img_scores[:, class_id]
                mask = scores > CONFIDENCE_THRESHOLD
                scores = scores[mask]
                if scores.size(0) == 0:
                    continue
                boxes = per_img_boxes[mask, :]
                boxes[:, 0:] *= self.height
                keep = boxes_nms(boxes, scores, NMS_THRESHOLD)
                # scores.sort(0, descending=True).values.cpu().numpy()
                # boxes[scores.sort(0, descending=True).indices].cpu().numpy()
                nmsed_boxes = boxes[keep, :]
                nmsed_labels = torch.tensor([class_id] * keep.size(0), device=device)
                nmsed_scores = scores[keep]
                processed_boxes.append(nmsed_boxes)
                processed_scores.append(nmsed_scores)
                processed_labels.append(nmsed_labels)

            if len(processed_boxes) == 0:
                processed_boxes = torch.empty(0, 2)
                processed_labels = torch.empty(0)
                processed_scores = torch.empty(0)
            else:
                processed_boxes = torch.cat(processed_boxes, 0)
                processed_labels = torch.cat(processed_labels, 0)
                processed_scores = torch.cat(processed_scores, 0)

            if processed_boxes.size(0) > MAX_PER_IMAGE > 0:
                processed_scores, keep = torch.topk(processed_scores, k=MAX_PER_IMAGE)
                processed_boxes = processed_boxes[keep, :]
                processed_labels = processed_labels[keep]
            results.append([processed_boxes, processed_labels, processed_scores])
        return results


def boxes_nms(boxes, scores, threshold=0.5):
        y1 = boxes[:, 0]
        y2 = boxes[:, 1]
        areas = (y2 - y1)  # [N,] 每个bbox的面积
        _, order = scores.sort(0, descending=True)  # 降序排列

        keep = []
        while order.numel() > 0:  # torch.numel()返回张量元素个数
            if order.numel() == 1:  # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()  # 保留scores最大的那个框box[i]
                keep.append(i)

            yy1 = y1[order[1:]]
            yy2 = y2[order[1:]]
            inter = (yy2 - yy1).clamp(min=0)  # [N-1,]

            iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
            idx = (iou <= threshold).nonzero().squeeze()    # 注意此时idx为[N-1,] 而order为[N,]
            if idx.numel() == 0:
                break
            order = order[idx + 1]  # 修补索引之间的差值
        return torch.LongTensor(keep)

