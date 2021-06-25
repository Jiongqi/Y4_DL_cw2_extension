import os
import torch
import torchvision
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dataloader import H5Dataset
from seg_network import Ensemble, loss_dice

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
filename = './dataset70-200.h5' # filename used in the dataloader

labeltype = 2 # segmentation sampling method used

## load saved classification and segmentation model
cla_filepath = './saved_models/New_Classification_model_params.pth'
seg_filepath = './saved_models/SEG_saved_segmentation_model_S{:01d}_params.pth'.format(labeltype)

cla_model = torchvision.models.vgg16(pretrained=False)
cla_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
cla_model.classifier[6] = nn.Linear(4096, 2, bias=True)
screen_seg_model = Ensemble()
noscreen_seg_model = Ensemble()

if use_cuda:
    cla_model.cuda(), screen_seg_model.cuda(), noscreen_seg_model.cuda()

cla_model.load_state_dict(torch.load(cla_filepath))
screen_seg_model.load_state_dict(torch.load(seg_filepath))
noscreen_seg_model.load_state_dict(torch.load(seg_filepath))



# testing data loader
test_set = H5Dataset(filename, network_type='segmentation', datatype='test', labeltype=labeltype)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1, # batch size set to 1 to allow loss calculate for each pair (ground truth, pred).
    shuffle=True, 
    num_workers=1)

## prepare for saving test metric (dice coefficient)
screen_metrics = []
noscreen_metrics = []

## prepare to save MSE 
criterion_mse = nn.MSELoss()
screen_mses = []
noscreen_mses = []

seg_threshold = 0.5 # segmentation threshold
samples = 100 # total test steps

cla_thresholds = np.arange(0,1.05,0.05) # classification thresholds used

sample = 0
while sample < samples:
    images, labels, case_id, frame_id = iter(test_loader).next()
    if use_cuda:
        images, labels, = images.cuda(), labels.cuda()
    with torch.no_grad():
        ## prepare to save noscreened images with predicted and ground truth labels
        Noscreen_overlaid_filename = './predict_results/Noscreen_overlaid_img_case{:04d}_frame{:03d}_method{:01d}.jpg'.format(int(case_id),int(frame_id),labeltype)
        if not os.path.exists(Noscreen_overlaid_filename):
            sample += 1

            # no screen segmentation model
            noscreen_preds = noscreen_seg_model(images)
            noscreen_binary_preds = torch.where(noscreen_preds<seg_threshold, 0, 1) # convert pred label images to binary image
            
            noscreen_loss = loss_dice(noscreen_binary_preds, labels)
            noscreen_metrics.append(1-noscreen_loss.item())
            noscreen_mse = criterion_mse(noscreen_binary_preds, labels)
            noscreen_mses.append(noscreen_mse.item())

            plt.figure()
            plt.imshow(images[0][0].cpu(), cmap='gray')
            plt.imshow(noscreen_binary_preds[0][0].cpu(), cmap='Blues', alpha=0.2)
            plt.imshow(labels[0][0].cpu(), cmap='Oranges', alpha=0.2)
            plt.axis('off')
            plt.savefig(Noscreen_overlaid_filename,bbox_inches='tight')
            plt.close()

            # screened (with the classification model) segmentation model
            feature_outputs = cla_model(images)
            prob_features = F.softmax(feature_outputs, dim=1)[0] ## convert classification output features in range [0,1]

            for cla_threshold in cla_thresholds:
                ## prepare to save screened images with different threshold (with pred and ground truth labels)
                Screen_threshold_overlaid_filename = './predict_results/Screen_threshold{:.0%}_overlaid_img_case{:04d}_frame{:03d}_method{:01d}_.jpg'.format(cla_threshold,int(case_id),int(frame_id),labeltype)
                class_output = 1 if prob_features[1]>cla_threshold else 0
                if class_output == 0:
                    screen_binary_preds = torch.zeros([1,1,64,64]).to(labels.device)
                else:
                    screen_preds = screen_seg_model(images)
                    screen_binary_preds = torch.where(screen_preds<seg_threshold, 0, 1)
                screen_loss = loss_dice(screen_binary_preds, labels)
                screen_metrics.append(1-screen_loss.item())
                screen_mse = criterion_mse(screen_binary_preds, labels)
                screen_mses.append(screen_mse.item())
                
                plt.figure()
                plt.imshow(images[0][0].cpu(), cmap='gray')
                plt.imshow(screen_binary_preds[0][0].cpu(), cmap='Greens', alpha=0.2)
                plt.imshow(labels[0][0].cpu(), cmap='Oranges', alpha=0.2)
                plt.axis('off')
                plt.savefig(Screen_threshold_overlaid_filename,bbox_inches='tight')
                plt.close()

            
'''
reshape screen metrics/mses:
    each row represents the metric/loss for the same image by using different thresholds
    each column represents the metric/loss for the same threshold by using different samples (tests)
'''
all_threshold_screen_metrics = np.array(screen_metrics).reshape(-1,len(cla_thresholds))
all_threshold_screen_mses = np.array(screen_mses).reshape(-1,len(cla_thresholds))

## saved losses
np.save('./data_for_plotting/All_threshold_Screen_test_metric.npy', all_threshold_screen_metrics)
np.save('./data_for_plotting/Nocreen_test_metric.npy', noscreen_metrics)
np.save('./data_for_plotting/MSE_All_threshold_Screen_test_metric.npy', all_threshold_screen_mses)
np.save('./data_for_plotting/MSE_Nocreen_test_metric.npy', noscreen_mses)
print('screen_metric',screen_metrics)
print('noscreen_metric',noscreen_metrics)
print('Testing done')