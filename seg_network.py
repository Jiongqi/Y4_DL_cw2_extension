import os
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

# from seg_architecture import UNet
from W2_new_seg_architecture import UNet
from dataloader import H5Dataset


os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
print('use_cuda:', use_cuda)


## loss function
def loss_dice(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=1, H, W]
    eps: smooth the loss to avoid zero denominator
    '''
    numerator = torch.sum(y_true*y_pred, dim=(2,3)) * 2
    denominator = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3)) + eps
    loss = torch.mean(1. - (numerator / denominator))
    return loss

## Bagging ensemble used (average three models' outputs)
class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.modelA = UNet(1,1)
        self.modelB = UNet(1,1)
        self.modelC = UNet(1,1)
        if use_cuda:
            self.modelA.cuda(), self.modelB.cuda(), self.modelC.cuda()

    def forward(self, x):
        preds1 = self.modelA(x)
        preds2 = self.modelB(x)
        preds3 = self.modelC(x)

        preds = (preds1 + preds2 + preds3)/3
        return preds




if __name__ == '__main__':

    labeltype = 1 # Segmentation label sampling method: 1 or 2
    seg_threshold = 0.5 # segmentation threshold
    filename = './dataset70-200.h5' # file name used as in data loader

    num_workers = 0
    batch_size = 8

    # training data loader
    train_set = H5Dataset(filename, network_type='segmentation', datatype='train', labeltype=labeltype)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    # validating data loader
    val_set = H5Dataset(filename, network_type='segmentation', datatype='val', labeltype=labeltype)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers)

    model = Ensemble()

    # optimisation loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    freq_print = 1#0   # in steps (frequency to print train loss)
    freq_val = 1#00  # in steps (frequency to run validation)
    total_steps = 10#000 

    train_losses = []
    val_metrics = []


    ## training
    step = 0
    while step < total_steps:
        for ii, (images, labels, _, _) in enumerate(train_loader):
            step += 1

            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            preds = model(images)
            loss = loss_dice(preds, labels)
            loss.backward()
            optimizer.step()

            # Compute and print loss (train)
            if (step % freq_print) == 0:    # print every freq_print mini-batches
                print('Step %d loss: %.5f' % (step,loss.item()))
                train_losses.append(loss.item())

            # Validation
            if (step % freq_val) == 0:  
                images_val, labels_val, _, _ = iter(val_loader).next()  # test one mini-batch
                if use_cuda:
                    images_val, labels_val = images_val.cuda(), labels_val.cuda()

                preds_val = model(images_val)
                preds_val_ = torch.where(preds_val<seg_threshold, 0,1)
                loss_val = loss_dice(preds_val_, labels_val)
                print('Step %d dice coefficient: %.5f' % (step,1-loss_val.item()))
                val_metrics.append(1-loss_val.item())

    ## saved location is subject to users (could be changed to any)    
    # np.save('./data_for_plotting/SEG_train_loss_method{:01d}.npy'.format(labeltype), train_losses)
    # np.save('./data_for_plotting/SEG_val_metric_method{:01d}.npy'.format(labeltype), val_metrics)
    print('Training done.')


    # save trained model (parameters)
    # torch.save(model.state_dict(), './saved_models/SEG_saved_segmentation_model_S{:01d}_params.pth'.format(labeltype))  
    # print('Model saved.')



