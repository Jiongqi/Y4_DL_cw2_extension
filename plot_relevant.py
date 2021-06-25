import matplotlib.pyplot as plt
import numpy as np

# function to get numerical values: mean, std, max, min
def calculate_numeric(data):
    numeric = []
    for i in data:
        mean = np.mean(i)
        std = np.std(i)
        max_value = np.max(i)
        min_value = np.min(i)
        numeric.append([mean, std, max_value, min_value])
    return numeric

# function to plot bland_altman_plot
def bland_altman_plot(data1, data2):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   
    mean_diff = np.mean(diff)                
    std_diff = np.std(diff, axis=0)           
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(mean, diff, color='orange')
    ax.axhline(0,color='gray', linestyle='--')
    ax.axhline(mean_diff + 1.96*std_diff, color='olivedrab', linestyle='--', label='mean difference + 1.96 STD')
    ax.axhline(mean_diff,color='gold', linestyle='--', label='mean difference')
    ax.axhline(mean_diff - 1.96*std_diff, color='cornflowerblue', linestyle='--', label='mean difference - 1.96 STD')    
    ax.set_xlabel('mean')
    ax.set_ylabel('difference')
    ax.legend()

## function for polyfit and plot the monitored train/val loss
def polyfit_monitored_loss(data1, data2, step1, step2, labels, degree=4):
    z1 = np.polyfit(step1, data1, degree)
    z2 = np.polyfit(step2, data2, degree)
    f1 = np.poly1d(z1)
    f2 = np.poly1d(z2)
    new_data1 = f1(step1)
    new_data2 = f2(step2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    

    ax.plot(step1, data1, alpha=0.1, color='red')
    ax.plot(step2, data2, alpha=0.1, color='blue')
    ax.plot(step1, new_data1, color='red', label=labels[0])
    ax.plot(step2, new_data2, color='blue', label = labels[1])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Train Loss')
    ax.legend()

###### ALL Datapath are subject to change depends on the name saved while running the testing processes.

## Question2: plot losses during training and validation for segmentation part
path_train1 = './data_for_plotting/SEG_train_loss_method1.npy'
path_train2 = './data_for_plotting/SEG_train_loss_method2.npy'
path_val1 = './data_for_plotting/SEG_val_metric_method1.npy'
path_val2 = './data_for_plotting/SEG_val_metric_method2.npy'
data_train1 = np.load(path_train1)
data_train2 = np.load(path_train2)
data_val1 = np.load(path_val1)
data_val2 = np.load(path_val2)

train_step = np.arange(0,10000, 10)
val_step = np.arange(0,10000, 100)

polyfit_monitored_loss(data_train1, data_train2, train_step, train_step, ['Segmentation label sampling 1','Segmentation label sampling 2'], 4)
plt.title('Training Loss')
polyfit_monitored_loss(data_val1, data_val2, val_step, val_step, ['Segmentation label sampling 1','Segmentation label sampling 2'], 3)
plt.title('Validation Loss')


# ## Question 5: compare segmentation label sampling methods
path1 = './data_for_plotting/SEG_test_metric_method1.npy'
path2 = './data_for_plotting/SEG_test_metric_method2.npy'
data1 = np.load(path1)
data2 = np.load(path2)

q5_numeric = calculate_numeric([data1, data2])

bland_altman_plot(data1, data2)
plt.title('Bland-Altman Plot: label sampling methods 1 VS 2') # could be adjusted according to requirements
plt.show()

## Question7: plot losses during training and validation for classification part
cla_path_train = './data_for_plotting/CLASS_train_loss.npy'
cla_path_val = './data_for_plotting/CLASS_val_metric.npy'
cla_data_train = np.load(cla_path_train)
cla_data_val = np.load(cla_path_val)

cla_train_step = np.arange(0,10000,10)
cla_val_step = np.arange(0,10000,100)

polyfit_monitored_loss(cla_data_train, cla_data_val, cla_train_step, cla_val_step, ['train loss','validation loss'],5)
plt.title('Monitoring during Training')

# ## Question 8: screen VS no screen; classificatoin threshold comparison
path_noscreen = './data_for_plotting/Nocreen_test_metric.npy'
path_screen = './data_for_plotting/All_threshold_Screen_test_metric.npy'
data_noscreen = np.load(path_noscreen)
data_screen = np.load(path_screen)

numeric_noscreen = calculate_numeric([data_noscreen])
threshold_x = np.arange(0,105,5)
for i in range(len(threshold_x)):
    numeric_screens = calculate_numeric([data_noscreen])
    print(numeric_screens, threshold_x[i])

bland_altman_plot(data_noscreen, data_screen[:,10]) # data_screen[:,10] corresponds to 50% threshold, the inputs can be adjusted according to need.
plt.title('Bland-Altman Plot (Dice Coefficient): No Screen VS With Screen')
plt.show()
