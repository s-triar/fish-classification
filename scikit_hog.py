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
# DATA_HOG_PATH = 'dataset/fish_data_hog'
# DATA_HL_PATH = 'dataset/fish_data_houghline'
# RADIUS_LBP = 2
# N_POINTS_LBP = 8 * RADIUS_LBP
# idx=0
# tt=0
# dd=0
# hh=0
# # with alive_bar(4415) as bar:
# for item in list(glob.glob(DATASET_PATH+'/*.*')):
#     idx+=1
#     filename = item.split('\\')[-1]
#     sub_filename = (filename.split('.')[0]).split('_')
#     class_name = '_'.join(sub_filename[:-1])
#     varian = sub_filename[-1]
#     img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img,(280,280))
    
#     # start_time   = time.time()    
#     # fd = hog(img, orientations=9, pixels_per_cell=(12, 12),
#     #             cells_per_block=(1, 1), visualize=False)
#     # elapsed_time = time.time() - start_time    
#         # f = open(DATA_HOG_PATH+'/'+'hog_fish.csv', 'a+', newline='', encoding='utf-8')
#         # writer = csv.writer(f)
#         # h = list(fd)
#         # h.insert(0,elapsed_time)
#         # h.insert(0,varian)
#         # h.insert(0,class_name)
#         # h.insert(0,idx)
#         # # id, class, varian, time, ...hog fts
#         # writer.writerow(h)
#         # f.close()
        
#     tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
#     # edges1 = canny(img)
    
#     start_time   = time.time()    
#     h, theta, d = hough_line(img, theta=tested_angles)
#     elapsed_time = time.time() - start_time    
#         # f = open(DATA_HL_PATH+'/'+'houghline_fish.csv', 'a+', newline='', encoding='utf-8')
#         # writer = csv.writer(f)
#         # fd = np.concatenate((theta,d),axis=None)
#         # h = list(fd)
#         # h.insert(0,elapsed_time)
#         # h.insert(0,varian)
#         # h.insert(0,class_name)
#         # h.insert(0,idx)
#         # # id, class, varian, time, ...theta fts (360) ,...d fts (793)
#         # writer.writerow(h)
#         # f.close()
        
#         # bar()
#     if idx%11==0 :
#         mse_h = (np.square(h, hh)).mean()
#         mse_theta = (np.square(theta, tt)).mean()
#         mse_d = (np.square(d, dd)).mean()
#         print(mse_d, mse_theta, mse_h)
#     if(idx%3==0):
#         tt=theta
#         dd=d
#         hh=h


DATASET_PATH='dataset/archive/Fish_Data/images/cropped'
item = os.path.join(DATASET_PATH,'A73EGS-P_4.png')
img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(img,(280,280))

tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(image, theta=tested_angles)
print(h.shape, theta.shape, d.shape)
print(image.shape)
fd, hog_image = hog(image, orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(1, 1), visualize=True)
# for i in fd:
#     print(i)
print(len(fd))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = hog_image #exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()

fix, axes = plt.subplots(1, 2, figsize=(7, 4))

axes[0].imshow(img, cmap=plt.cm.gray)
axes[0].set_title('Input image')

angle_step = 0.5 * np.rad2deg(np.diff(theta).mean())
d_step = 0.5 * np.diff(d).mean()
bounds = (np.rad2deg(theta[0]) - angle_step,
          np.rad2deg(theta[-1]) + angle_step,
          d[-1] + d_step, d[0] - d_step)

axes[1].imshow(h, cmap=plt.cm.bone, extent=bounds)
axes[1].set_title('Hough transform')
axes[1].set_xlabel('Angle (degree)')
axes[1].set_ylabel('Distance (pixel)')

plt.tight_layout()
plt.show()


# item = os.path.join(DATASET_PATH,'A73EGS-P_4.png.png')
# img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(img,(280,280))

# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
# edges1 = canny(image)
# h2, theta2, d2 = hough_line(edges1, theta=tested_angles)
# print(h2.shape, theta2.shape, d2.shape)
# print(image.shape)
# fd, hog_image = hog(image, orientations=9, pixels_per_cell=(12, 12),
#                     cells_per_block=(1, 1), visualize=True)
# # for i in fd:
# #     print(i)
# print(len(fd))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')

# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()

# fix, axes = plt.subplots(1, 2, figsize=(7, 4))

# axes[0].imshow(edges1, cmap=plt.cm.gray)
# axes[0].set_title('Input image')

# angle_step = 0.5 * np.rad2deg(np.diff(theta2).mean())
# d_step = 0.5 * np.diff(d2).mean()
# bounds = (np.rad2deg(theta2[0]) - angle_step,
#           np.rad2deg(theta2[-1]) + angle_step,
#           d2[-1] + d_step, d2[0] - d_step)

# axes[1].imshow(h2, cmap=plt.cm.bone, extent=bounds)
# axes[1].set_title('Hough transform')
# axes[1].set_xlabel('Angle (degree)')
# axes[1].set_ylabel('Distance (pixel)')

# plt.tight_layout()
# plt.show()

# mse_theta = (np.square(theta, theta2)).mean()
# mse_d = (np.square(d, d2)).mean()

# print(mse_d,mse_theta)

# # Initiate FAST detector
# star = cv2.xfeatures2d.StarDetector_create()
# # Initiate BRIEF extractor
# brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# # find the keypoints with STAR
# kp = star.detect(image,None)
# # compute the descriptors with BRIEF
# kp, des = brief.compute(image, kp)
# print( brief.descriptorSize() )
# print( des.shape, des )
# image2 = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
# plt.imshow(image2)
# plt.show()
# # Initiate ORB detector
# orb = cv2.ORB_create(nfeatures=4,patchSize=20)
# # find the keypoints with ORB
# kp = orb.detect(image,None)
# # compute the descriptors with ORB
# kp, des = orb.compute(image, kp)
# print( brief.descriptorSize() )
# print( des.shape, des )
# # draw only keypoints location,not size and orientation
# image2 = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
# plt.imshow(image2)
# plt.show()