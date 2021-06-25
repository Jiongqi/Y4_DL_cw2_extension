import os
import torchvision
import torch
from torch import nn
import numpy as np

from dataloader import H5Dataset

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()

filename = './dataset70-200.h5' # filename used in data loaders

## load the original VGG16 model
model = torchvision.models.vgg16(pretrained=True)

## blocked all weights first, changes on structure later will be assigned by dafult with param.requires_grad = True
for param in model.parameters():
    param.requires_grad = False

## change the input channel in the first layer from 3 to 1
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
## change the last layer's output feature from 1000 to 2
model.classifier[6] = nn.Linear(4096, 2, bias=True)

if use_cuda:
    model.cuda()

num_workers = 0
batch_size = 8
# training data loader
train_set = H5Dataset(filename, network_type='classification', datatype='train', labeltype=2)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)

# validating data loader
val_set = H5Dataset(filename, network_type='classification', datatype='val', labeltype=2)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers)


criterion = torch.nn.CrossEntropyLoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# optimisation loop
freq_print = 1#0   # in steps, frequency to print train loss
freq_val = 1#00  # in steps, frequency to run validation
cl_total_steps = 10#000

train_losses = []
val_metrics = []


## training
step = 0
while step < cl_total_steps:
    for ii, (frames, cl_labels,_,_) in enumerate(train_loader):
        step+=1
        if use_cuda:
            frames, cl_labels = frames.cuda(), cl_labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(frames)
        outputs.long()
        loss = criterion(outputs, cl_labels)
        loss.backward()
        optimizer.step()
        
        # Compute and print loss
        if (step % freq_print) == 0:    # print every freq_print mini-batches
            print('step %d - train loss: %.3f' % (step, loss.item()))
            train_losses.append(loss.item())
            

        # Validation
        if (step % freq_val) == 0:  
            frames_val, cl_labels_val, _, _ = iter(val_loader).next()  # test one mini-batch
            if use_cuda:
                frames_val, cl_labels_val = frames_val.cuda(), cl_labels_val.cuda()
            outputs_val = model(frames_val)
            loss_val = criterion(outputs_val, cl_labels_val)
            print('step %d - val loss: %.3f' % (step, loss_val.item()))
            val_metrics.append(loss_val.item())

# np.save('./data_for_plotting/CLASS_train_loss.npy', train_losses)
# np.save('./data_for_plotting/CLASS_val_metric.npy', val_metrics)
print('Training done.')


## save trained model (parameters)
# torch.save(model.state_dict(), './saved_models/Classification_model_params.pth')  
# print('Model saved.')


