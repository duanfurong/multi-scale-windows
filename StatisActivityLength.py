import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from week16.label import ground_truth


def get_ActivityLength():
    length1 = 0
    length2 = 0
    length3 = 0
    length4 = 0
    length5 = 0
    length_list = []
    for label_name in ground_truth.keys():
        groundValue_list = ground_truth[label_name]['gt_boxes']
        for value in groundValue_list:
            length = value[1] - value[0]
            if length <= 10:
                length1 += 1
            elif length > 10 and length <= 30:
                length2 +=1
            elif length > 30 and length <= 100:
                length3 += 1
            else:
                length4 +=1
    length_list.append(length1)
    length_list.append(length2)
    length_list.append(length3)
    length_list.append(length4)
    print('length1:' + str(length1))
    print('length2:' + str(length2))
    print('length3:' + str(length3))
    print('length4:' + str(length4))
    plot_bar(length_list)
        # for ground_value in label_name.keys():
        #     ground_value['']

def plot_bar(data):
    labels = ['less than 10', '10 and 30', '30 and 100', '100 and 200']
    plt.title('Wisdom data set length distribution')
    plt.bar(range(len(data)), data, tick_label=labels)
    plt.show()
get_ActivityLength()
