from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import time
import glob
import csv
from alive_progress import alive_bar # for indicator process

# Declare path to dataset
DATASET_PATH='dataset/archive/Fish_Data/images/cropped'

# Declare available methods in local_binary_pattern
# from https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern
METHODS = ['default', 'ror','uniform','nri_uniform','var']

# Declare path to target folder to save the feature.
# Create folder manually
DATA_LBP_PATH = 'dataset/fish_data_lbp/resize/'

# Set the radius of kernel used in lbp later
RADIUS_LBP = 1
N_POINTS_LBP = 8 * RADIUS_LBP

# declare index for numbering
idx=0

# process inside alive_bar
with alive_bar(4412) as bar:
    # looping each image in dataset folder
    for item in list(glob.glob(DATASET_PATH+'/*.*')):
        # increment index for numbering
        idx+=1
        
        # get the class and varian
        filename = item.split('\\')[-1]
        sub_filename = (filename.split('.')[0]).split('_')
        class_name = '_'.join(sub_filename[:-1])
        varian = sub_filename[-1]
        
        # read image as grayscale and resize
        img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(280,280))
        
        # looping over available methods
        for m in METHODS:
            # set start time process
            start_time   = time.time()    
            # get lbp feature by calling library
            temp = local_binary_pattern(img, N_POINTS_LBP, RADIUS_LBP, m)
            # calculate the time it took to process
            elapsed_time = time.time() - start_time    
            # generate the lbp into histogram
            hist = np.histogram(temp, density=True, bins=256, range=(0, 256))
            # write the data into a file
            f = open(DATA_LBP_PATH+'/'+m+'/'+m+'_dense'+'_radius_is_5.csv', 'a+', newline='', encoding='utf-8')
            writer = csv.writer(f)
            h = list(hist[0])
            h.insert(0,elapsed_time)
            h.insert(0,varian)
            h.insert(0,class_name)
            h.insert(0,idx)
            # write the data according columns below
            # id, class, varian, time, ...lbp hist
            writer.writerow(h)
            f.close()
        bar()