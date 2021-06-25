import torch
import numpy as np
from torch import nn
import h5py

train_file_path = 'train.h5'
train_file = h5py.File(train_file_path, 'r')

dataset = train_file
# index = 425
# labels = np.array([dataset[k] for k in dataset.keys() if k.split('_')[0]=='label' and int(k.split('_')[1])==index])
# sum_labels = np.sum(labels, axis=0)
# consensus_label = np.where(sum_labels<1*2, 0, 1)
# consensus_label = torch.unsqueeze(torch.tensor(consensus_label), axis=0).float()
# print(labels.shape)
a = np.array([k for k in dataset.keys() if k.split('_')[0]=='frame'])# and int(k.split('_')[1])==index])
print(a)


