import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# sensor_names = ['AccX','AccY', 'AccZ', 'GraX', 'GraY', 'GraZ', 'MagX', 'MagY', 'MagZ']
sensor_names = ['Acc_X', 'Acc_Y', 'Acc_Z']


train_examples = []
valid_examples = []


class ActivityDataSet(Dataset):

    def __init__(self, root_dir):
        data_list = []
        name_list = []
        length_list = []
        self.root_dir = root_dir
        _, _, files = list(os.walk(root_dir))[0]
        for file in files:
            data = pd.read_csv(os.path.join(root_dir, file))
            name = file.split('_')[1].split('.')[0]
            # data.drop('timestamp', axis=1, inplace=True)
            for sensor in sensor_names:
                data[sensor] = moving_average(data[sensor].values)
            data_list.append(data[sensor_names].values)
            name_list.append(name)
            length_list.append(len(data))
        self.name_list = name_list
        self.data_list = data_list
        self.length = len(data_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        if index > self.length - 1: raise Exception('index is too big!')
        data = self.data_list[index]
        name = self.name_list[index]
        return torch.from_numpy(data), name

    def __len__(self):
        return self.length



def moving_average(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


