import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def plot_Curve(data, color, label, ylabel):
    plt.plot(data, color=color, label=label)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks([])
    plt.show()

def plot_Bare(title,data,barName,color):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title)
    activityF1Score = ('write on notepad', 'open hood', 'close hood', ' open left front door', 'close left front door')
    buy_number = [0.85, 0.89, 0.90, 0.92, 0.91]
    # ,,'LightBLue'
    # plt.bar(activityF1Score, buy_number, color=['orange', 'blue', 'green', 'yellow', 'SteelBlue'])
    plt.bar(barName, data, color=color)
    plt.show()

def plot_confusion(comfusion,class_data):
    plt.figure(figsize=(12,9))
    plt.rcParams['font.family'] = ['Times New Roman']
    classes = class_data
    plt.imshow(comfusion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.title('confusion_matrix',fontsize = 12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=315)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=12)
    thresh = comfusion.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (comfusion.size, 2))
    for i, j in iters:
        plt.text(j, i, format(comfusion[i, j]),verticalalignment="center",horizontalalignment="center")  # 显示对应的数字

    plt.ylabel('Real label',fontsize = 12)
    plt.xlabel('Predicted label',fontsize = 12)

    plt.tight_layout()
    plt.savefig('wisdm_confusion.png')
    # plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap='Blues',  # 这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 7))
    #    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=15)
    plt.xlabel('Predicted label', size=15)
    caption = str(accuracy) + '_' + title + '_Confusion.png'
    plt.savefig('./Confusion/' + caption, format='png', bbox_inches='tight')
    # plt.show()
