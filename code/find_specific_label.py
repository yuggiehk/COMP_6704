
import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import glob
import cv2
from PIL import Image
import copy
import shutil
from skimage.io import imsave

train_label_folder = '/mnt/nvme1/suyuejiao/egohos_split_data/train/label/'
test_indomain_label_folder = '/mnt/nvme1/suyuejiao/egohos_split_data/test_indomain/label/'
test_outdomain_label_folder = '/mnt/nvme1/suyuejiao/egohos_split_data/test_outdomain/label/'

train_image_folder = '/mnt/nvme1/suyuejiao/egohos_split_data/train/image/'

list = []

# new_folder_img = '/mnt/nvme1/suyuejiao/egohos_split_data/samples2/sam_img/'
# new_folder_lbl = '/mnt/nvme1/suyuejiao/egohos_split_data/samples2/sam_lbl/'

# for file in tqdm(glob.glob(train_label_folder+'*.png')):
#     print(file)
#     filename = file.split('/')[-1].split('.')[0]
#     label = np.array(Image.open(file))
#     uni_labl = np.unique(label)
#     if 5 in uni_labl and (3 in uni_labl and 4 in uni_labl):
#         print(True)
#         list.append(file)
#         shutil.copyfile(train_image_folder+filename+'.jpg', new_folder_img+filename+'.jpg')
#         shutil.copyfile(file, new_folder_lbl+filename+'.png')

# print(list)

# new_folder_vis = '/mnt/nvme1/suyuejiao/egohos_split_data/samples2/sam_vis/'

def visualize_twohands_obj2(img, seg_result, alpha = 0.6):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    # seg_color[seg_result == 1] = (255,  0,   0)     # left_hand
    # seg_color[seg_result == 2] = (0,    0,   255)   # right_hand
    # seg_color[seg_result == 3] = (255,  0,   255)   # left_object1
    seg_color[seg_result == 4] = (0,    255, 255)   # right_object1
    seg_color[seg_result == 5] = (0,    255, 255)     # two_object1
    vis = img * (1 - alpha) + seg_color * alpha
    return vis

# for file in tqdm(glob.glob(new_folder_lbl+'*.png')):
#     lbl = np.array(Image.open(file))
#     filename = file.split('/')[-1].split('.')[0]
#     img = np.array(Image.open(new_folder_img+filename+'.jpg'))

#     vis = visualize_twohands_obj2(img, lbl)
#     imsave(new_folder_vis+filename+'.jpg', vis.astype(np.uint8))



img_path = '/mnt/nvme1/suyuejiao/egohos_split_data/samples2/sam_img/ego4d_def2e8dd-aaf7-467f-aa8f-46f654e6f4e0_45300.jpg'
lbl_path = '/mnt/nvme1/suyuejiao/egohos_split_data/samples2/sam_lbl/ego4d_def2e8dd-aaf7-467f-aa8f-46f654e6f4e0_45300.png'


# lbl = np.array(Image.open(lbl_path))
# img = np.array(Image.open(img_path))
# vis = visualize_twohands_obj2(img,lbl)
# imsave('/mnt/nvme1/suyuejiao/egohos_split_data/samples2'+'3.jpg', vis.astype(np.uint8))

cb_path = '/mnt/nvme1/suyuejiao/egohos_split_data/train/label_contact_first/ego4d_def2e8dd-aaf7-467f-aa8f-46f654e6f4e0_45300.png'
def visualize_cb(img, seg_result, alpha = 0.6):
    seg_color = np.zeros((img.shape))
    seg_color[seg_result == 0] = (0,    0,   0)     # background
    seg_color[seg_result == 1] = (255,  255,   0)     # contact 
    vis = img * (1 - alpha) + seg_color * alpha
    return vis
cb = np.array(Image.open(cb_path))
img = np.array(Image.open(img_path))
vis_cb = visualize_cb(img,cb)
imsave('/mnt/nvme1/suyuejiao/egohos_split_data/samples2'+'5.jpg', vis_cb.astype(np.uint8))