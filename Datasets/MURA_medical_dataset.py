# https://www.kaggle.com/ahmadsayed/model-inception

from __future__ import print_function, division
import time
from tqdm import tqdm
from random import shuffle
import keras
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import itertools
import os
from keras.preprocessing import image



def MURA_train_path(name):
    train_paths = []
    for name in names:
        label = None
        arr = name.split("/")
        if 'positive' in arr[4]:
            root = shoulder_path_train
            label = 0
        elif 'negative' in arr[4]:
            root = shoulder_path_train
            label = 1
        
        root = root + "/" + arr[3] + "/" + arr[4] + "/" + arr[5]
        train_paths.append([root, label])
     return train_paths


def getImageArr(path, size):
    try:
        bgr = cv2.imread(path)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = cv2.resize(bgr, (size, size))
        #img = np.divide(img, 255)
        return img
    except Exception as e:
        #img = np.zeros((size, size, 3))
        pass

# make dataset paths
img_dir='/Users/downloads/MURA-v1.1/train'
path_dir='/Users/downloads/'
shoulder_path_train =os.path.join(img_dir,  "XR_SHOULDER")

names = []
f = open(os.path.join(path_dir, "train_image_paths.csv"))
for row in f:
    names.append(row.strip())


# save normal images
train_paths = MURA_train_path(name)
normal =[]
for i, v in train_paths:
    if v==0:
        img = getImageArr(i, 256)
        if img is not None:
            normal.append(img)
            
print(len(normal)) # 4559
plt.imshow(normal[1])
plt.show()

save_dir ='/Users/downloads/normal_medical/'
for i, im in enumerate(normal):
    img = image.array_to_img(im, scale=False)
    img.save(os.path.join(save_dir, 'medical_{}.png'.format(i)))



# save anomaly images
anomaly =[]
for i, v in train_paths:
    if v==1:
        img = getImageArr(i, 256)
        if img is not None:
            anomaly.append(img)
len(anomaly)



save_dir ='/Users/downloads/ano_medical/'
for i, im in enumerate(anomaly):
    img = image.array_to_img(im, scale=False)
    img.save(os.path.join(save_dir, 'ano_medical_{}.png'.format(i)))
