import torch
import csv
import numpy as np
import pandas as pd
from train import DataLoad_Process

path_model = './model/model'+str(2)+'.pkl'
net_load = torch.load(path_model)
val = pd.read_csv('./datasets/gender_submission.csv')
val_sur = val['Survived']
ri_count = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data,test = DataLoad_Process()
for i, data_tuple in enumerate(test, 0):
    test_d = data_tuple[0]
    test_d = test_d.to(device)
    net_load = net_load.to(device)
    out = net_load(test_d)
    out = 1 if out>0.5 else 0
    if out == val_sur[i]:
        ri_count += 1
items = val.shape[0]
print('acc:',ri_count/items)
