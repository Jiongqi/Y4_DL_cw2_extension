import os
import torch
import torchvision
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dataloader import H5Dataset

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
filename = './dataset70-200.h5'

## load original VGG16 model and update the weight based on the saved trained classification model
filepath = './saved_models/Classification_model_params.pth'
model = torchvision.models.vgg16(pretrained=False)
model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
model.classifier[6] = nn.Linear(4096, 2, bias=True)

if use_cuda:
    model.cuda()
model.load_state_dict(torch.load(filepath))


# testing data loader
test_set = H5Dataset(filename, network_type='classification', datatype='test', labeltype=2)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1,
    shuffle=True, 
    num_workers=1)

losses = []
accum_results = []
accum_true_labels = []

threshold = 0.5 # classification threshold
samples = 100 # total test steps


criterion = torch.nn.CrossEntropyLoss(reduction='mean')


sample = 0
while sample < samples:
    images, labels, case_id, frame_id = iter(test_loader).next()
    if use_cuda:
        images, labels = images.cuda(), labels.cuda()
    with torch.no_grad():
        image_with_class_filename = './Cal_predict_labels/Cla_case{:04d}_frame{:03d}_label.jpg'.format(int(case_id),int(frame_id))
        if not os.path.exists(image_with_class_filename):
            sample += 1

            cl_outputs = model(images) # output: features, format -> [prob1, prob2]
            loss = criterion(cl_outputs, labels)
            losses.append(loss.item())

            prob = F.softmax(cl_outputs, dim=1)[0] # return probs in cl_outputs in range [0,1]
            cl_predict_labels = 1 if prob[1]>threshold else 0 # convert to a class number according to the threshold set
            accum_results.append(cl_predict_labels) # saved all predict class results
            accum_true_labels.append(labels) # saved corresponding ground truth class results

            # saved original image with case id, frame id, ground truth label and predict label
            plt.figure()
            plt.imshow(images[0][0].cpu(), cmap='gray')
            plt.axis('off')
            plt.title('Cases{}, Frame{}, Consensus Label{}, Predict Label{}'.format(int(case_id), int(frame_id), int(labels), cl_predict_labels))
            plt.savefig(image_with_class_filename,bbox_inches='tight')
            plt.close()

## calculate classification accuracy
def calculate_accuracy(preds, truth):
    diff =  np.array(truth) - np.array(preds)
    wrong = np.count_nonzero(diff)
    return ((len(diff)-wrong) / len(diff))

accuracy = calculate_accuracy(accum_results, accum_true_labels)
print('accuracy:{:.0%}'.format(accuracy))
np.save('./data_for_plotting/Cla_test_loss.npy', losses)
print('losses',losses)
print('Testing done.')