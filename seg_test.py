import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from dataloader import H5Dataset
from seg_network import loss_dice, Ensemble

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()

filename = './dataset70-200.h5' # filename used for the data loader

## filepath to load the saved model (parameters)
filepath1 = './saved_models/SEG_saved_segmentation_model_S1_params.pth'
filepath2 = './saved_models/SEG_saved_segmentation_model_S2_params.pth'
model1 = Ensemble()
model2 = Ensemble()
if use_cuda:
    model1.cuda(), model2.cuda()
model1.load_state_dict(torch.load(filepath1))
model2.load_state_dict(torch.load(filepath2))


# testing data loader
test_set = H5Dataset(filename, network_type='segmentation', datatype='test', labeltype=2)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1, # batch size set to 1 to avoid loss function is calculated for each pair of images
    shuffle=True, 
    num_workers=1)

## prepared to save test metric (dice coefficient)
metrics1 = []
metrics2 = []

## saved MSE loss
criterion_mse = nn.MSELoss()
mses1 = []
mses2 = []

threshold = 0.5 # segmentation threshold to get label image in binary

samples = 100 # total test steps runs

sample = 0
while sample < samples:
    images, labels, case_id, frame_id = iter(test_loader).next()
    if use_cuda:
        images, labels = images.cuda(), labels.cuda()
    with torch.no_grad():
        ## save predict labels' images and overlaid images (for two segmentation label sampling methods)
        pred_label1_filename = './result_images/SEG_predlabel_case{:04d}_frame{:03d}_method1.jpg'.format(int(case_id),int(frame_id))
        pred_label2_filename = './result_images/SEG_predlabel_case{:04d}_frame{:03d}_method2.jpg'.format(int(case_id),int(frame_id))
        overlaid_image1_filename = './result_images/SEG_overlaid_img_case{:04d}_frame{:03d}_method1.jpg'.format(int(case_id),int(frame_id))
        overlaid_image2_filename = './result_images/SEG_overlaid_img_case{:04d}_frame{:03d}_method2.jpg'.format(int(case_id),int(frame_id))
        if not os.path.exists(overlaid_image1_filename):
            sample += 1
            
            preds1 = model1(images)
            preds2 = model2(images)
            
            ## convert predict labels image to binary images
            pred_label1 = torch.where(preds1<threshold, 0, 1)
            pred_label2 = torch.where(preds2<threshold, 0, 1)

            ## (loss, metric) -> (dice loss, dice coefficient); mse -> MSE loss
            loss1 = loss_dice(pred_label1, labels)
            loss2 = loss_dice(pred_label2, labels)
            metrics1.append(1-loss1.item())
            metrics2.append(1-loss2.item())
            mse1 = criterion_mse(pred_label1, labels)
            mse2 = criterion_mse(pred_label2, labels)
            mses1.append(mse1.item())
            mses2.append(mse2.item())

            ## save images 
            plt.figure()
            plt.imshow(pred_label1[0][0].cpu(), cmap='gray')
            plt.axis('off')
            plt.savefig(pred_label1_filename,bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.imshow(pred_label2[0][0].cpu(), cmap='gray')
            plt.axis('off')
            plt.savefig(pred_label2_filename,bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.imshow(images[0][0].cpu(), cmap='gray')
            plt.imshow(pred_label1[0][0].cpu(), cmap='Greens', alpha=0.2)
            plt.imshow(labels[0][0].cpu(), cmap='Oranges', alpha=0.2)
            plt.axis('off')
            plt.savefig(overlaid_image1_filename,bbox_inches='tight')
            plt.close()
                            
            plt.figure()
            plt.imshow(images[0][0].cpu(), cmap='gray')
            plt.imshow(pred_label2[0][0].cpu(), cmap='Greens', alpha=0.2)
            plt.imshow(labels[0][0].cpu(), cmap='Oranges', alpha=0.2)
            plt.axis('off')
            plt.savefig(overlaid_image2_filename,bbox_inches='tight')
            plt.close()

np.save('./data_for_plotting/TTSEG_test_metric_method1.npy', metrics1)
np.save('./data_for_plotting/TTSEG_test_metric_method2.npy', metrics2)
np.save('./data_for_plotting/SEG_test_mse_method1.npy', mses1)
np.save('./data_for_plotting/SEG_test_mse_method2.npy', mses2)
## print if needed
# print('SEG_test_metric_method1',metrics1)
# print('SEG_test_metric_method2',metrics2)
print('Testing done.')