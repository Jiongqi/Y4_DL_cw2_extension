import random
import h5py
import torch
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
import torchvision.transforms.functional as TF

# Data augmentation
def data_augmentation(frame, label):
    # Random horizontal flipping
    if random.random() > 0.5:
        frame = TF.hflip(frame)
        label = TF.hflip(label)

    # Random vertical flipping
    if random.random() > 0.5:
        frame = TF.vflip(frame)
        label = TF.vflip(label)

    return frame, label


# data loader
class H5Dataset(torch.utils.data.Dataset):
    '''
    network_type: 'segmentation' or 'classification' 
    datatype: 'train' or 'val' or 'test'
    labeltype --> Segmentation label sampling method: 1 or 2 
    train_valtest_ratio: ratio between train and val&test cases, val&test cases are separated again equally
     *e.g. train_valtest_ratio=[0.7,0.3] means 70% cases are usd in training, 15% for validation and 15% for testing.
    '''
    def __init__(self, file_path, network_type, datatype, labeltype=2, train_valtest_ratio=[0.7,0.3]):
        super(H5Dataset, self).__init__()
        if network_type is 'classification' and labeltype is 1:
            raise ValueError('labeltype=2 -> The classification model is deisgned based on the 2nd segmentation sampling method.')
        self.network_type = network_type
        self.datatype = datatype
        self.labeltype = labeltype
        self.h5_file = h5py.File(file_path, 'r')
        self.num_labels = len(set([k.split('_')[3] for k in self.h5_file.keys() if k.split('_')[0]=='label']))
        self.total_num_cases = len(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.train_num_cases = int(self.total_num_cases*train_valtest_ratio[0])
        self.valtest_num_cases = int(self.total_num_cases*train_valtest_ratio[1]/2)

    def __len__(self):
        return (self.train_num_cases if self.datatype is 'train' else self.valtest_num_cases)

    def segmentation_label_sampling_2(self, idx_frame, index):
        labels = np.array([self.h5_file[k] for k in self.h5_file.keys() if k.split('_')[0]=='label' and int(k.split('_')[1])==index and k.split('_')[2]==idx_frame])
        sum_labels = np.sum(labels, axis=0)
        consensus_label = np.where(sum_labels<1*2, 0, 1)
        consensus_label = torch.unsqueeze(torch.tensor(consensus_label), axis=0).float()
        return consensus_label

    def __getitem__(self, index): ## index -> case id (int); idx_frame -> frame id (str).
        if self.datatype is 'val':
            index+=self.train_num_cases
        if self.datatype is 'test':
            index+=(self.train_num_cases+self.valtest_num_cases)
        
        # randomly choose a frame from a case
        frames_nums = [k.split('_')[2] for k in self.h5_file.keys() if k.split('_')[0]=='frame' and int(k.split('_')[1])==index]
        idx_frame = random.choice(frames_nums)
        frame = torch.unsqueeze(torch.tensor(self.h5_file['frame_%04d_%s' % (index, idx_frame)][()].astype('float32')), dim=0)

        ## get label based on the select segmentation label sampling method
        if self.network_type is 'segmentation' and self.labeltype ==1:
            idx_label = random.randint(0, self.num_labels-1)
            label = torch.tensor(self.h5_file['label_%04d_%s_%02d' % (index, idx_frame, idx_label)][()].astype('float32')).unsqueeze(0)
        else:
            label = self.segmentation_label_sampling_2(idx_frame, index)


        # increase size to [64,64] to avoid issue caused by odd image size being processed in training.
        frame = torch.squeeze(F.interpolate(frame.unsqueeze(0), size=[64,64]), dim=0)
        label = torch.squeeze(F.interpolate(label.unsqueeze(0), size=[64,64]), dim=0)
        
        if self.datatype is 'train':
            ## data augmentation
            frame, label = data_augmentation(frame, label)
        
        if self.network_type is 'segmentation':
            return frame, label, index, int(idx_frame)
        else:
            cl_label = torch.tensor(1) if label.mean()>0 else torch.tensor(0) # return single tensor as a class: 0 or 1
            return frame, cl_label, index, int(idx_frame)

            


