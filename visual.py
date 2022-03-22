from calendar import day_abbr
from fileinput import filename
# from skimage.transform import rotate
from skimage.feature import local_binary_pattern
# from skimage import data
import numpy as np
# from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2
import time
import glob
RADIUS_LBP = 1
N_POINTS_LBP = 8 * RADIUS_LBP
DATASET_PATH='dataset/archive/Fish_Data/images/cropped'
item = DATASET_PATH+'/'+'A73EGS-P_4.png'
img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(280,280))
lbp5 = local_binary_pattern(img, N_POINTS_LBP, RADIUS_LBP, 'default')
RADIUS_LBP = 1
N_POINTS_LBP = 8 * RADIUS_LBP
lbp1 = local_binary_pattern(img, N_POINTS_LBP, RADIUS_LBP, 'default')

# cv2.imshow("input",img)
# cv2.imshow("lbp_r=5",lbp5)
# cv2.imshow("lbp_r=1",lbp1)
# cv2.waitKey(0)
# plt.figure()
# plt.imshow(img)
# plt.colorbar()
# plt.grid(False)
# plt.show()

plt.hist(lbp5)
plt.show()