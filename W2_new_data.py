import SimpleITK as sitk
import numpy as np

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

path_0 = r'C:\Users\QJQ\Desktop\trus_uint8\images\VOL_1417174641772_FRAMES.nii'
data_0 = read_img(path_0)

path_1 = r'C:\Users\QJQ\Desktop\trus_uint8\seg_01\label_VOL_1417174641772_FRAMES.nii'
data_1 = read_img(path_1)

path_2 = r'C:\Users\QJQ\Desktop\trus_uint8\seg_02\label_VOL_1417174641772_FRAMES.nii'
data_2 = read_img(path_2)

path_3 = r'C:\Users\QJQ\Desktop\trus_uint8\seg_03\label_VOL_1417174641772_FRAMES.nii'
data_3 = read_img(path_3)

print(data_0.shape)
print(data_0.type)
print(data_3.shape)
print(data_3.type)