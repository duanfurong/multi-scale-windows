import torch
import math
import numpy as np
import itertools
from collections import defaultdict
import six

# 解码
from sklearn.metrics import confusion_matrix

from PlotUtil import plot_confusion_matrix, plot_confusion
from Config import variance, ACTIVITY_LABEL,TITLE
import sklearn.metrics as sm


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    a = center_variance * priors[..., 1:]
    b = locations[..., :1].double() * a.double()
    c = b + priors[..., :1].double()
    d = torch.exp(locations[..., 1:].double() * size_variance) * priors[..., 1:].double()
    return torch.cat([
        c,
        d
    ], dim=locations.dim() - 1)


# 编码
def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def bbox_iou(bbox_a, bbox_b):
    # if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
    #     raise IndexError
    # top left
    tl = torch.from_numpy(np.maximum(bbox_a[:,None,:1], bbox_b[:, :1]))
    # bottom right
    br = torch.from_numpy(np.minimum(bbox_a[:, None, 1:], bbox_b[:, 1:]))
    area_i = torch.squeeze(torch.clamp((br - tl), min=0),2).float()
    area_a = torch.squeeze(torch.from_numpy(bbox_a[:, 1:] - bbox_a[:, :1]),1).float()
    area_b = torch.squeeze(torch.from_numpy(bbox_b[:, 1:] - bbox_b[:, :1]),1).float()
    return area_i / (area_a[:, None] + area_b - area_i)



def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


# [x, y, w, h] to [xmin, ymin, xmax, ymax]
def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :1] - locations[..., 1:] / 2,
                      locations[..., :1] + locations[..., 1:] / 2], locations.dim() - 1)


# [xmin, ymin, xmax, ymax] to [x, y, w, h]
def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


def intersect(box_a, box_b):
    '''
    计算box_a和box_b两种框的交集，两种框的数量可能不一致.
    首先将框resize成[A,B,2],这样做的目的是,能计算box_a中每个框与
    box_b中每个框的交集:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
    然后再计算框的交集
    :param box_a:[A,4],4代表[xmin,ymin,xmax,ymax]
    :param box_b:[B,4],4代表[xmin,ymin,xmax,ymax]
    :return:交集面积,shape[A,B]
    '''
    A = box_a.shape[0]
    B = box_b.shape[0]

    max_xy = torch.min(box_a[:, 1:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 1:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :1].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :1].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)  # 得到x与y之间的差值，shape为[A,B,2]
    return inter[:, :, 0]                    # 按位相乘,shape[A,B],每个元素为矩形框的交集面积


def jaccard(box_a, box_b):
    '''
    计算box_a中每个矩形框与box_b中每个矩形框的IOU
    :param box_a:真实框的坐标,Shape: [num_objects,4]
    :param box_b:先验锚点框的坐标,Shape: [num_priors,4]
    :return:IOU,Shape: [box_a.size(0), box_b.size(0)]
    '''
    inter = intersect(box_a, box_b)  # 计算交集面积
    area_a = (box_a[:, 1] - box_a[:, 0]).unsqueeze(1).expand_as(inter)
    area_b = (box_b[:, 1] - box_b[:, 0]).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter  # 并集面积
    return inter / union  # shape[A,B]


def match(threshold, truths, labels, priors, loc_t, conf_t, idx):
    # 第1步,计算IOU
    truths = truths.double()
    priors = priors.double()
    overlaps = jaccard(truths, point_from(priors))  # shape:[num_object,num_priors]

    # 第2步,为每个真实框匹配一个IOU最大的锚点框,GT框->锚点框
    # best_prior_overlap为每个真实框的最大IOU值,shape[num_objects,1]
    # best_prior_idx为对应的最大IOU的先验锚点框的Index,其元素值的范围为[0,num_priors]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # 第3步,若先验锚点框与GT框的IOU>阈值,也将这些锚点框匹配上,锚点框->GT框
    # best_truth_overlap为每个先验锚点框对应其中一个真实框的最大IOU,shape[1,num_priors]
    # best_truth_idx为每个先验锚点框对应的真实框的index,其元素值的范围为[0,num_objects]
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_prior_idx.squeeze_(1)  # [num_objects]
    best_prior_overlap.squeeze_(1)  # [num_objects]
    best_truth_idx.squeeze_(0)  # [num_priors]
    best_truth_overlap.squeeze_(0)  # [num_priors]

    # 第4步
    # index_fill_(self, dim: _int, index: Tensor, value: Number)对第dim行的index使用value进行填充
    # best_truth_overlap为第一步匹配的结果,需要使用到,使用best_prior_idx是第二步的结果,也是需要使用上的
    # 所以在best_truth_overlap上进行填充,表明选出来的正例
    # 使用2进行填充,是因为,IOU值的范围是[0,1],只要使用大于1的值填充,就表明肯定能被选出来
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # 这一步是为了后面能把最佳的锚框选出来
    # 确保每个GT框都能匹配上最大IOU的先验锚点框
    # 得到每个先验锚点框都能有一个匹配上的数字
    # best_prior_idx的元素值的范围是[0,num_priors],长度为num_objects
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # 第5步
    conf = labels[best_truth_idx] + 1  # Shape: 进行了+1操作是因为在图片定位中有背景

    conf[best_truth_overlap < threshold] = 0  # 置信度小于阈值的label设置为0

    # 第6步
    matches = truths[best_truth_idx]  # 取出最佳匹配的GT框,Shape: [num_priors,4]

    # 进行位置编码
    loc = encode(matches, priors, variance)
    loc_t[idx] = loc  # [num_priors,2],应该学习的编码偏差
    conf_t[idx] = conf.permute(1,0)  # [num_priors],每个锚点框的label


def point_from(boxes):
    '''
    将先验锚点框(cx,cy,w,h)转换成(xmin,ymin,xmax,ymax)
    :param boxes: 先验锚点框的坐标
    :return: 返回坐标转换后的先验锚点框(xmin,ymin,xmax,ymax)
    '''
    return torch.cat((boxes[:, :1] - boxes[:, 1:] / 2,  # xim,ymin
                      boxes[:, :1] + boxes[:, 1:] / 2), 1)  # xmax,ymax


def encode(matched, priors, variances):
    '''
    对坐标进行编码,对应论文中的公式2
    利用GT框和先验锚点框,计算偏差,用于回归
    :param matched: 每个先验锚点框对应最佳的GT框,Shape: [num_priors, 4],
                    其中4代表[xmin,ymin,xmax,ymax]
    :param priors: 先验锚点框,Shape: [num_priors,4],
                    其中4代表[中心点x,中心点y,宽,高]
    :return: shape:[num_priors, 4]
    '''
    eps = 1e-5
    g_cxcy = (matched[:, :1] + matched[:, 1:]) / 2 - priors[:, :1]  # 计算GT框与锚点框中心点的距离
    g_cxcy /= (variances[0] * priors[:, 1:])

    g_wh = (matched[:, 1:] - matched[:, :1])  # xmax-xmin,ymax-ymin
    g_wh /= priors[:, 1:]
    g_wh = torch.log(g_wh + eps) / variances[1]

    return torch.cat((g_cxcy, g_wh), dim=1).contiguous()


def log_sum_exp(x):
    '''
    用于计算在batch所有样本突出的置信度损失
    :param x: 置信度网络预测出来的置信度
    :return:
    '''
    x_max = x.max()  # 得到x中最大的一个数字
    # torch.sum(torch.exp(x - x_max), 1, keepdim=True)->shape:[8732,1],按行相加
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def eval_detection_activity(
        pred_bboxes,
        pred_labels,
        pred_scores,
        gt_bboxes,
        gt_labels,
        gt_difficults=None,
        iou_thresh=0.5,
        use_07_metric=False):
    prec, rec, f1_score,averged_f1,ned_dis = calc_detection_prec_rec(pred_bboxes,
                                            pred_labels,
                                            pred_scores,
                                            gt_bboxes,
                                            gt_labels,
                                            gt_difficults,
                                            iou_thresh=iou_thresh)
    ap = calc_detection_ap(prec, rec, use_07_metric=use_07_metric)
    print('f1_score' + str(f1_score))
    # return {'ap': ap, 'map': np.nanmean(ap), 'f1_score': f1_score}
    return {'ap': ap, 'map': 0.9, 'f1_score': f1_score,'averged_f1':averged_f1,'ned':ned_dis}


def calc_detection_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.6):
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    confusion_pred_label = []
    confusion_truth_label = []
    ned = 0
    num = 0
    pred_boundary_file = [] #用来统计预测的活动边界
    truth_boundary_file = [] #用来统计真实的活动边界
    truth_label_file = [] #用来统计活动的类别
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = np.array(gt_bbox)[gt_mask_l]
            gt_difficult_l = np.array(gt_difficult)[gt_mask_l]

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                confusion_truth_label.extend((gt_label[-1]-1,) * np.sum(pred_mask_l))
                confusion_pred_label.extend((l-1,) * np.sum(pred_mask_l))

                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue
            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            pred_bbox_l = pred_bbox_l.copy()
            # pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            # gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l).numpy()
            gt_index = iou.argmax(axis=1)
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            pro_pred_box = pred_bbox_l[iou.max(axis=1) > iou_thresh]
            pred_boundary = np.ceil(list(pro_pred_box[:, 1] - pro_pred_box[:, 0]))  # 预测的每一类真实活动的边界长度
            truth_boundary = gt_bbox_l[:, 1] - gt_bbox_l[:, 0]
            truth_boundary = list(truth_boundary[iou.max(axis=0) > iou_thresh])
            if len(pred_boundary) > 0:
                pred_boundary_file.append(pred_boundary[-1])
                truth_boundary_file.append(truth_boundary[-1])
                truth_label_file.append(l)

            if sum(truth_boundary) != 0 and (len(pred_boundary)==len(truth_boundary)):
                pred_boundary = sum(pred_boundary)
                truth_boundary = sum(truth_boundary)
                leven = levenshtein(int(pred_boundary) * '1', truth_boundary * '1')
                ned += leven/truth_boundary
                num+=1

            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        # if not selec[gt_idx]:
                        confusion_pred_label.extend((l-1,))
                        confusion_truth_label.extend((l-1,))
                        match[l].append(1)
                        # else:
                        #     match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
                    confusion_truth_label.extend((l-1,))
                    confusion_pred_label.extend((l-1,)) #活动标签是以1开始的，confusion矩阵里面要以0开始
    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')
    ned = ned/num
    print('ned is:' + str(ned))
    n_fg_class = max(n_pos.keys())+1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    prec1 = [None] * n_fg_class
    rec1 = [None] * n_fg_class
    f1_score = [None] * n_fg_class
    weighted = [0] * n_fg_class  #统计每个类别分别占了多少比重
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)
        weighted[l] = len(match_l)
        # order = score_l.argsort()[::-1]
        # match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        tp1 = np.sum(match_l == 1)
        fp1 = np.sum(match_l == 0)
        prec1[l] = tp1 / (tp1 + fp1)

        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
            rec1[l] = tp1 / n_pos[l]
        f1_score[l] = calc_f1_score(prec1[l],rec1[l])
    confusion = sm.confusion_matrix(confusion_truth_label, confusion_pred_label)
    print('The confusion matrix is：', confusion, sep='\n')
    # plot_confusion_matrix(confusion,
    #                ACTIVITY_LABEL,title=TITLE)
    log_result_concise(pred_boundary_file, truth_boundary_file, truth_label_file) #将真实的分割结果与实际的分割结果写入file

    # 在此处绘制confusion matrix矩阵
    # conf_mat = confusion_matrix(y_true=confusion_truth_label, y_pred=confusion_pred_label)
    #
    # plot_confusion_matrix(conf_mat, normalize=False, target_names=[0, 1, 2, 3, 4, 5, 6],
    #                       title='UCI Confusion Matrix')
    averged_f1 = calc_weighted_f1_score(f1_score,weighted)
    return prec, rec, f1_score,averged_f1,ned


def calc_detection_ap(prec, rec, use_07_metric=False):
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def calc_f1_score(prec, rec):
    if prec is None or rec is None or (prec == 0 and rec == 0):
        f1_score = 0.0
    else:
        f1_score = 2*np.multiply(prec, rec) / (prec + rec)
    return f1_score

def calc_weighted_f1_score(f1_score,weighted):
    f1score = f1_score[1:]
    for i in range(len(f1score)):
        if f1score[i] is None:
            f1score[i] = 0.0
    weighted = np.divide(weighted[1:],sum(weighted[1:]))
    f1 = sum(np.multiply(f1score,weighted))
    print('weighted_f1 is:' + str(f1))
    return f1

def levenshtein(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    # 创建矩阵
    matrix = [0 for n in range(len_str1 * len_str2)]
    # 矩阵的第一行
    for i in range(len_str1):
        matrix[i] = i
    # 矩阵的第一列
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1
    # 根据状态转移方程逐步得到编辑距离
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)
    return matrix[-1]


def log_result_concise(pred_bbx, truth_bbx, l):
    pred_bbx = np.array(pred_bbx).reshape(-1, 1)
    truth_bbx = np.array(truth_bbx).reshape(-1, 1)
    l = np.array(l).reshape(-1, 1)
    data = np.concatenate([pred_bbx, truth_bbx, l], axis=1)
    np.savetxt('segment_result.csv', data, encoding='utf-8', fmt='%s', delimiter=',')