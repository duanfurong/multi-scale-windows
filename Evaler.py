import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Boxs_op import eval_detection_activity
from Config import ACTIVITY_SIZE
from LabelConfig import eval_ground_target
from PostProcess import postprocessor


class Evaler(object):

    def __init__(self, eval_devices=None):
        self.postprocessor = postprocessor()
        self.eval_devices = eval_devices
        self.file_handle = open('result.txt', mode='a+')
        self.file_score_handle = open('averged_score.txt', mode='a+')
        self.file_ned_handle = open('ned_score.txt', mode='a+')

    def __call__(self, model, test_dataset):
        test_loader = DataLoader(dataset=test_dataset, batch_size=1600)
        results_dict = self.eval_model_inference(model, data_loader=test_loader)
        result = cal_ap_map(results_dict)
        ap, map, f1_score,averged_f1,ned_dis = result['ap'], result['map'], result['f1_score'], result['averged_f1'],result['ned']
        self.file_handle.writelines(str(f1_score) + '\n')
        self.file_score_handle.writelines(str(averged_f1) + '\n')
        self.file_ned_handle.writelines(str(ned_dis) + '\n')
        return ap, map, f1_score

    def eval_model_inference(self, model, data_loader):
        with torch.no_grad():
            results_dict = {}
            print(' Evaluating...... use GPU : {}'.format(self.eval_devices))
            for batch in tqdm(data_loader):
                data, name = batch
                data = data.to(self.eval_devices)
                bbox_pred, cls_logits, priors = model.forward_with_testprocess(data)
                results = self.postprocessor(cls_logits, bbox_pred, priors)
                for num_name, result in zip(name, results):
                    pred_boxes, pred_labels, pred_scores = result
                    pred_boxes, pred_labels, pred_scores = pred_boxes.to('cpu').numpy(), \
                                                           pred_labels.to('cpu').numpy(), \
                                                           pred_scores.to('cpu').numpy()
                    results_dict.update({num_name: {'pred_boxes': pred_boxes,
                                                    'pred_labels': pred_labels,
                                                    'pred_scores': pred_scores}})
        return results_dict


def cal_ap_map(results_dict):
    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxs_list = []
    gt_labels_list = []
    gt_difficult_list = []
    for num_name in results_dict:
        gt_boxs, gt_labels, gt_difficult = eval_ground_target[num_name]['gt_boxes'], eval_ground_target[num_name]['gt_label'], eval_ground_target[num_name]['gt_diffcult']
        h = ACTIVITY_SIZE  # 数据的高度
        pred_boxes, pred_labels, pred_scores = results_dict[num_name]['pred_boxes'], results_dict[num_name][
            'pred_labels'], results_dict[num_name]['pred_scores']
        pred_boxes[:, 0:] *= (h / ACTIVITY_SIZE)
        pred_boxes_list.append(pred_boxes)
        pred_labels_list.append(pred_labels)
        pred_scores_list.append(pred_scores)
        gt_boxs_list.append(gt_boxs)
        gt_labels_list.append(gt_labels)
        gt_difficult_list.append(gt_difficult)
    result = eval_detection_activity(pred_bboxes=pred_boxes_list,
                                     pred_labels=pred_labels_list,
                                     pred_scores=pred_scores_list,
                                     gt_bboxes=gt_boxs_list,
                                     gt_labels=gt_labels_list,
                                     gt_difficults=gt_difficult_list)
    return result
