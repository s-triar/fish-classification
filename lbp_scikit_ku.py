from fileinput import filename
# from skimage.transform import rotate
from skimage.feature import local_binary_pattern
# from skimage import data
import numpy as np
# from skimage.color import label2rgb
# import matplotlib.pyplot as plt
import cv2
import time
import glob
# import os
import csv
from alive_progress import alive_bar

DATASET_PATH='dataset/archive/Fish_Data/images/cropped'
METHODS = ['default'] #'ror','uniform','nri_uniform','var'
DATA_LBP_PATH = 'dataset/fish_data_lbp/resize/'
RADIUS_LBP = 5
N_POINTS_LBP = 8 * RADIUS_LBP
idx=0
with alive_bar(4412) as bar:
    for item in list(glob.glob(DATASET_PATH+'/*.*')):
        idx+=1
        filename = item.split('\\')[-1]
        sub_filename = (filename.split('.')[0]).split('_')
        class_name = '_'.join(sub_filename[:-1])
        varian = sub_filename[-1]
        img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(280,280))
        for m in METHODS:
            start_time   = time.time()    
            temp = local_binary_pattern(img, N_POINTS_LBP, RADIUS_LBP, m)
            elapsed_time = time.time() - start_time    
            hist = np.histogram(temp, density=True, bins=256, range=(0, 256))
            f = open(DATA_LBP_PATH+'/'+m+'/'+m+'_dense'+'_radius_is_5.csv', 'a+', newline='', encoding='utf-8')
            writer = csv.writer(f)
            h = list(hist[0])
            h.insert(0,elapsed_time)
            h.insert(0,varian)
            h.insert(0,class_name)
            h.insert(0,idx)
            # id, class, varian, time, ...lbp hist
            writer.writerow(h)
            f.close()
            # hist = np.histogram(temp, density=False, bins=256, range=(0, 256))
            # f = open(DATA_LBP_PATH+'/'+m+'/'+m+''+'_radius_is_2.csv', 'a+', newline='', encoding='utf-8')
            # writer = csv.writer(f)
            # h = list(hist[0])
            # h.insert(0,elapsed_time)
            # h.insert(0,varian)
            # h.insert(0,class_name)
            # h.insert(0,idx)
            # # id, class, varian, time, ...lbp hist
            # writer.writerow(h)
            # f.close()
        bar()