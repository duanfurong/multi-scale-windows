import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

from PlotUtil import plot_confusion_matrix
from Boxs_op import match, log_sum_exp

from Config import ACTIVITY_SIZE


class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes, overlap_thresh, neg_pos):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes  # 类别数
        self.threshold = overlap_thresh  # GT框与先验锚点框的阈值
        self.negpos_ratio = neg_pos  # 正负样本的比例
        self.activity_height = ACTIVITY_SIZE

    def forward(self, prediction, targets):
        bbox_pred, scores, boxes = prediction
        num = bbox_pred.shape[0]  # 得到有多少个batch
        num_priors = boxes.squeeze().shape[0]  # 总共有多少个锚框
        loc_t = torch.Tensor(num, num_priors, 2)  # [batch_size,1232,2],生成随机tensor,最后面得到匹配后的位置标签
        conf_t = torch.Tensor(num, num_priors)  # 得到匹配后的类别标签
        for idx in range(num):
            ground_Boxes = np.array(targets[idx]['gt_boxes'])
            groundTruths = torch.Tensor(ground_Boxes / ACTIVITY_SIZE).cuda()
            ground_label = targets[idx]['gt_label']
            groundLabels = torch.Tensor(ground_label).cuda()  # label
            defaults = boxes.squeeze().cuda()  # 默认框
            match(self.threshold, groundTruths, groundLabels, defaults, loc_t, conf_t, idx)
        if torch.cuda.is_available():
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()  # shape:[batch_size,8732],其元素组成是类别标签号和背景
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        # compute_aps(conf.shape[0], conf, best_truth_overlap, 0.5)
        pos = conf_t > 0  # 排除label=0,即排除背景,shape[batch_size,8732],其元素组成是true或者false
        # Localization Loss (Smooth L1),定位损失函数
        # Shape: [batch,num_priors,4]
        # pos.dim()表示pos有多少维,应该是一个定值(2)
        # pos由[batch_size,8732]变成[batch_size,8732,1],然后展开成[batch_size,8732,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(bbox_pred)
        loc_p = bbox_pred[pos_idx].view(-1, 2).double()  # [num_pos,4],取出带目标的这些框
        loc_t = loc_t[pos_idx].view(-1, 2).double()  # [num_pos,4]
        # loc_t = torch.DoubleTensor([[1770.4839, 903.3871, 28.6105, 28.6105]]).cuda()
        # 位置损失函数
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')  # 这里对损失值是相加,有公式可知,还没到相除的地步
        # 为Hard Negative Mining计算max conf across batch
        batch_conf = scores.squeeze().view(-1, self.num_classes)  # shape[batch_size*8732,5]
        # gather函数的作用是沿着定轴dim(1),按照Index(conf_t.view(-1, 1))取出元素
        # batch_conf.gather(1, conf_t.view(-1, 1))的shape[8732,1],作用是得到每个锚点框在匹配GT框后的label
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1).long())  # 这个不是最终的置信度损失函数
        # Hard Negative Mining
        # 由于正例与负例的数据不均衡,因此不是所有负例都用于训练
        loss_c[pos.view(-1, 1)] = 0  # pos与loss_c维度不一样,所以需要转换一下,选出负例 将正样本的loss置为0
        loss_c = loss_c.view(num, -1)  # [batch_size,8732]
        _, loss_idx = loss_c.sort(1, descending=True)  # 得到降序排列的index
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)  # pos里面是true或者false,因此sum后的结果应该是包含的目标数量
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  # 生成一个随机数用于表示负例的数量,正例和负例的比例约3:1
        neg = idx_rank < num_neg.expand_as(idx_rank)  # [batch_size,8732] 选择num_neg个负例,其元素组成是true或者false
        # 置信度损失,包括正例和负例
        # [batch_size, 8732, 21],元素组成是true或者false,但true代表着存在目标,其对应的index为label
        pos_idx = pos.unsqueeze(2).expand_as(scores)
        neg_idx = neg.unsqueeze(2).expand_as(scores)
        # pos_idx由true和false组成,表示选择出来的正例,neg_idx同理
        # (pos_idx + neg_idx)表示选择出来用于训练的样例,包含正例和反例
        # torch.gt(other)函数的作用是逐个元素与other进行大小比较,大于则为true,否则为false
        # 因此conf_data[(pos_idx + neg_idx).gt(0)]得到了所有用于训练的样例
        conf_p = scores[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted.long(), reduction='sum')
        length = conf_p.shape[0]
        total_accuracy = (torch.argmax(F.softmax(conf_p, dim=1),dim=1) == targets_weighted.long()).sum().float().item() / length
        # plot_matrix(torch.argmax(F.softmax(conf_p, dim=1), dim=1).cpu().numpy(), targets_weighted.long().cpu().numpy())
        back_accy = ((torch.argmax(F.softmax(conf_p, dim=1), dim=1) == 0) & (targets_weighted.long() == 0)).sum()
        back_accy = back_accy.item() / (targets_weighted.long() == 0).sum().item()
        first_accy = ((torch.argmax(F.softmax(conf_p, dim=1), dim=1) == 1) & (targets_weighted.long() == 1)).sum()
        first_accy = first_accy.item() / (targets_weighted.long() == 1).sum().item() if (targets_weighted.long() == 1).sum().item() > 0 else 0
        second_accy = ((torch.argmax(F.softmax(conf_p, dim=1), dim=1) == 2) & (targets_weighted.long() == 2)).sum()
        second_accy = second_accy.item() / (targets_weighted.long() == 2).sum().item() if (targets_weighted.long() == 2).sum().item() > 0 else 0
        third_accy = ((torch.argmax(F.softmax(conf_p, dim=1), dim=1) == 3) & (targets_weighted.long() == 3)).sum()
        third_accy = third_accy.item() / (targets_weighted.long() == 3).sum().item() if (targets_weighted.long() == 3).sum().item() > 0 else 0
        fourth_accy = ((torch.argmax(F.softmax(conf_p, dim=1), dim=1) == 4) & (targets_weighted.long() == 4)).sum()
        fourth_accy =  fourth_accy.item() / (targets_weighted.long() == 4).sum().item() if (targets_weighted.long() == 4).sum().item() > 0 else 0
        print(
            'total_accy : {} | back_accy : {} |first_acc : {} | second_acc : {:.4f} | third_acc : {:.4f} | forthd_acc : {:.4f}'.format(
                total_accuracy, back_accy,
                first_accy, second_accy, third_accy, fourth_accy))
        # L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.sum()  # 一个batch里面所有正例的数量
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c, total_accuracy, back_accy, first_accy, second_accy, third_accy, fourth_accy


def plot_matrix(pred_label,true_label):
    conf_mat = confusion_matrix(y_true=true_label, y_pred=pred_label)

    plot_confusion_matrix(conf_mat, normalize=False, target_names=[0, 1, 2, 3, 4, 5, 6, 7],
                              title='Confusion Matrix')