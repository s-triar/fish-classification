from fileinput import filename
from skimage.transform import rotate
from skimage.feature import local_binary_pattern, hog
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

import sys
from time import time

import numpy as np
import matplotlib.pyplot as plt

from dask import delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
# 

DATASET_PATH='dataset/archive/Fish_Data/images/cropped'
item = os.path.join(DATASET_PATH,'zeus_faber_5.png')
img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(img,(280,280))

@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)


# To speed up the example, extract the two types of features only
feature_types = ['type-2-x', 'type-2-y']

# Build a computation graph using Dask. This allows the use of multiple
# CPU cores later during the actual computation
ii = integral_image(image)

# Compute the result
t_start = time()
X = haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_types,
                             feature_coord=None)
X = np.array(X)
time_full_feature_comp = time() - t_start

print(X)
print(time_full_feature_comp)
print(X.shape)
