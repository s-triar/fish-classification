from fileinput import filename
from skimage.transform import rotate
from skimage.feature import local_binary_pattern, hog, canny
from skimage.exposure import equalize_hist
from skimage.transform import hough_line, hough_line_peaks
from skimage import data
import numpy as np
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2
import time
import glob
import os
import csv
from alive_progress import alive_bar
from skimage import data, exposure


DATASET_PATH='dataset/archive/Fish_Data/images/cropped'
DATA_HOG_PATH = 'dataset/fish_data_hog'
DATA_HL_PATH = 'dataset/fish_data_houghline'
RADIUS_LBP = 1
N_POINTS_LBP = 8 * RADIUS_LBP
idx=0
tt=0
dd=0
hh=0
with alive_bar(4415) as bar:
    for item in list(glob.glob(DATASET_PATH+'/*.*')):
        idx+=1
        filename = item.split('\\')[-1]
        sub_filename = (filename.split('.')[0]).split('_')
        class_name = '_'.join(sub_filename[:-1])
        varian = sub_filename[-1]
        img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(280,280))
        
        start_time   = time.time()    
        fd = hog(img, orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(1, 1), visualize=False)
        elapsed_time = time.time() - start_time    
        f = open(DATA_HOG_PATH+'/'+'hog_fish.csv', 'a+', newline='', encoding='utf-8')
        writer = csv.writer(f)
        h = list(fd)
        h.insert(0,elapsed_time)
        h.insert(0,varian)
        h.insert(0,class_name)
        h.insert(0,idx)
        # id, class, varian, time, ...hog fts
        writer.writerow(h)
        f.close()
        
        # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        # edges1 = canny(img)
        # start_time   = time.time()    
        # h, theta, d = hough_line(edges1, theta=tested_angles)
        # elapsed_time = time.time() - start_time    
        # f = open(DATA_HL_PATH+'/'+'houghline_fish.csv', 'a+', newline='', encoding='utf-8')
        # writer = csv.writer(f)
        # fd = np.concatenate((theta,d),axis=None)
        # h = list(fd)
        # h.insert(0,elapsed_time)
        # h.insert(0,varian)
        # h.insert(0,class_name)
        # h.insert(0,idx)
        # # id, class, varian, time, ...theta fts (360) ,...d fts (793)
        # writer.writerow(h)
        # f.close()
        
        bar()
