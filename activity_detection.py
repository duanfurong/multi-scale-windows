import torch
import numpy as np
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader
from ActivityDataSet import ActivityDataSet
from ActivitySSD import ActivitySSD
from Config import n_epochs, clip, model_path
from Evaler import Evaler
import matplotlib.pyplot as plt
#
# train_datasets = ActivityDataSet('datasets/train')
# eval_datasets = ActivityDataSet('datasets/eval')
#
# train_loader = DataLoader(train_datasets, batch_size=200, shuffle=True)
# eval_loader = DataLoader(eval_datasets, batch_size=200, shuffle=True)
# test_loader = DataLoader(test_datasets, batch_size=8, shuffle=True)

# learning_rate = 5e-1
learning_rate = 0.001
# weight_decay = 5e-4
weight_decay = 0.0001
momentum = 0.9
ssd = ActivitySSD()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
params = [p for p in ssd.parameters() if p.requires_grad]
# optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay,
#                       momentum=momentum)
optimizer = optim.Adam(params, lr=0.001)
lscheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

best_f1score = 0.0
best_ap1 = 0.0
best_ap2 = 0.0
best_ap3 = 0.0
best_ap4 = 0.0


def train_detectActivity(root_dir):
    ssd.train()
    ssd.to(device)
    train_loss = []
    cls_loss = []
    reg_loss = []
    lr_rate = []
    train_datasets = ActivityDataSet(root_dir+'/train')
    train_loader = DataLoader(train_datasets, batch_size=210, shuffle=True)
    test_datasets = ActivityDataSet(root_dir+'/test')
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_loader):
            data, name = batch
            data = data.to(device)
            loss_l, loss_c, class_accuracy, back_accuracy, first_accuracy, second_accuracy, third_accuracy, fourth_accuracy = ssd.forward_with_postprocess(
                data, name)
            loss = 3*loss_c + 1 * loss_l
            train_loss.append(loss.item())
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(ssd.parameters(), clip)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # lr_scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            lr_rate.append(lr)
            if epoch % 200 == 0:
                cls_loss.append(loss_c.item())
                reg_loss.append(loss_l.item())

            if epoch % 5 == 0:
                print('Iter : {}/{} | Lr : {} | Loss : {:.4f} | cls_loss : {:.4f} | reg_loss : {:.4f}'.format(epoch,
                                                                                                              n_epochs,
                                                                                                              lr,
                                                                                                              loss.item(),
                                                                                                              loss_c.item(),
                                                                                                              loss_l.item()))
        ssd.eval()
        evaler = Evaler(eval_devices=device)
        # 验证器开始在数据集上验证模型
        ap, map, f1_score = evaler(model=ssd,
                         test_dataset=test_datasets)
        # avg_f1score = np.mean(f1_score[1:])
        # print('avg_f1score:'+str(avg_f1score))
        # if avg_f1score >= best_f1score:
        #     best_f1score = avg_f1score
        #     # torch.save(ssd.state_dict(), model_path, _use_new_zipfile_serialization=False)
        #     torch.save(ssd.state_dict(), model_path)
    plt.plot(cls_loss, color='orange', label='cls_loss')
    plt.plot(reg_loss, color='red', linewidth=1, label='reg_loss')
    plt.xlabel("epoch per 200")
    plt.ylabel('loss')
    plt.title("learning rate is " + str(learning_rate))
    plt.legend()
    plt.show()



# def get_Kfold_data(root_dir, num_folds=5):
#         load_all_files(root_dir)
#         kf = KFold(n_splits=num_folds, shuffle=True)
#         train_data_arr = np.array(data_list)
#         for train_index, val_index in kf.split(train_data_arr):
#             train_datasets = ActivityDataSet('train', train_index)
#             eval_datasets = ActivityDataSet('train', val_index)
#             train_detectActivity(train_datasets, eval_datasets)



# def test_detectActivity():
#     test_datasets = ActivityDataSet()
#     ssd.load_state_dict(torch.load(model_path))
#     evaler = Evaler(eval_devices=device)
#     # 验证器开始在数据集上验证模型
#     ssd.eval()
#     ap, map = evaler(model=ssd,
#                      test_dataset=test_datasets)
#     print('eval ap is' + str(ap))
#     print('eval map is' + str(map))


if __name__ == '__main__':
    # get_Kfold_data(root_dir='datasets/train')
    train_detectActivity(root_dir='./datasets')

    # test_detectActivity()
