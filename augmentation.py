from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import time
import glob
import csv
from alive_progress import alive_bar # for indicator process
import os
import random
import pandas as pd
import math

# Declare path to dataset
DATASET_PATH='dataset/archive/Fish_Data/images/cropped'

# Declare available methods in local_binary_pattern
# - ori : only resize and conver into 3 channel if the image is 1 channel
# - sharpen : sharp the image
# - gaus-blur : do the gaussian blur
# - brigthness : tweak the image brightness 
# - contrast : tweak the image contrast
# - combi : tweak the image contrast and brightness
# - hist-eq : do the histogram equalization on the image
# - color-cast : add or reduce the pixel's value with a value
AUGMENTATION_TYPE = ['ori','sharpen', 'gaus-blur','brightness','contrast','hist-eq','combi','color-cast']

# Declare path to target folder to save the augmentation data.
DATA_SAVE_PATH = 'dataset/augmentation/'
# create the target folder respectively with the augmentation method
if not os.path.exists(DATA_SAVE_PATH):
    os.makedirs(DATA_SAVE_PATH)
for i in AUGMENTATION_TYPE:
    if not os.path.exists(os.path.join(DATA_SAVE_PATH,i)):
        os.makedirs(os.path.join(DATA_SAVE_PATH,i))

# define a method for sharping the image
def sharpen(img):
    kernel = np.array([
        [ 0, 0,-1, 0, 0],
        [ 0,-1,-2,-1, 0],
        [-1,-2,15,-2, 1],
        [ 0,-1,-2,-1, 0],
        [ 0, 0,-1, 0, 0]
    ])
    return cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

# define a method for doing gaussian blur
def g_blur(img):
    return cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)

# define a method to do brightness and/or contrast
def _apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# define a method for doing brightness and/or constras (brightness, contrast, combi)
def brigtness_and_contrast(img, needMore=0):
    blist_for_brigtness = [-127, 127] # list of brightness values for brightness augmentation 
    clist_for_brigtness = [   0,   0] # list of contrast values for brightness augmentation
    blist_for_contrast =  [   0,   0] # list of brightness values for contrast augmentation
    clist_for_contrast =  [ -64,  64] # list of contrast values for contrast augmentation
    brigtnesses = []
    contrasts = []
    for idx in range(2):
        b = _apply_brightness_contrast(img,blist_for_brigtness[idx],clist_for_brigtness[idx])
        brigtnesses.append(b)
        c = _apply_brightness_contrast(img,blist_for_contrast[idx],clist_for_contrast[idx])
        contrasts.append(c)
    additional=[]
    for i in range(needMore):
        bsetting = random.randint(-50,50) # limit the range to -50 to 50
        csetting = random.randint(-50,50) # limit the range to -50 to 50
        a = _apply_brightness_contrast(img,bsetting,csetting)
        additional.append(a)
    return (brigtnesses,contrasts,additional)

# define a method for casting the image's color
def color_clasting(img, needMore=0):
    res = []
    for i in range(needMore):
        imgcopy=img.copy()
        channels = ['r','g','b','rg','rb','gb','rgb']
        chooseChannel = random.randint(0,len(channels)-1)
        valueChanger = random.randint(-40,40) # limit to -40 to 40  value change
        if('r' in channels[chooseChannel]):
            imgcopy[:,:,0] = cv2.add(imgcopy[:,:,0],valueChanger)
        if('g' in channels[chooseChannel]):
            imgcopy[:,:,1] = cv2.add(imgcopy[:,:,1],valueChanger)
        if('b' in channels[chooseChannel]):
            imgcopy[:,:,2] = cv2.add(imgcopy[:,:,2],valueChanger)
        res.append(imgcopy)
    return res
    
# define a method for doing histogram equalization
def hist_eq(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    

# read the class data
df = pd.read_csv('dataset/archive/Fish_Data/final_all_index.txt', header=None, sep='=')
df[5] = df.apply(lambda x: x[3].split("_")[-1], axis=1)
# count member of each class
dfg = df.groupby(df[1])[0].count()

# set the number of the target desire. +8 because there are 8 augmentation default.
# the remaining value is using combi and color-cast
TARGET_BALANCE = 40 - 8 
# declare index for numbering
idx=0
print(len(df))
print(df.head(10))
print(dfg)

buffer={}

for i in range(len(dfg)):
    print(dfg.index[i])

# # process inside alive_bar
# with alive_bar(4412) as bar:
#     # looping each image in dataset folder
# for item in range(len(df)):
#     # increment index for numbering
#     idx+=1
        
#     # get the class and varian
#     # filename = item.split('\\')[-1]
#     # sub_filename = (filename.split('.')[0]).split('_')
#     class_name = df.iloc[item][1] #'_'.join(sub_filename[:-1])
#     print(class_name)
#     varian = df.iloc[item][5] #sub_filename[-1]
#     print(varian)
    # indexInDF = np.where(dfg.index.values == class_name)
    # nClass = dfg[indexInDF[0][0]]
    # needMore = math.floor(((TARGET_BALANCE*1.0)-nClass) / nClass) 
    # b_c_more = math.ceil(needMore/2)
    # clr_cst_more = needMore - b_c_more
    # check_n = ((b_c_more+clr_cst_more)*nClass)+nClass
    # if( check_n < TARGET_BALANCE):
    #     if(df.iloc[item][5] == nClass):
    #         red = TARGET_BALANCE - check_n
    #         adt = math.ceil((red*1.0)/2)
    #         b_c_more += adt
    #         clr_cst_more+=adt
        
    if(idx>10):
        break
    # # read image as grayscale and resize
    # img = cv2.imread(os.path.join(DATASET_PATH, df.iloc[item][3]+'.png'), cv2.IMREAD_UNCHANGED)
    # img = cv2.resize(img,(100,100))
    # if(len(img.shape)==2):
    #     img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # # augmentation
    # # ['ori', 'sharpen', 'gaus-blur','brightness','contrast','hist-eq','combi','color-cast']
    
    # # do the augmentation
    # sharp = sharpen(img) 
    # gaus_blur = g_blur(img)
    # (brigts,contrs, combi) = brigtness_and_contrast(img,b_c_more)
    # cc = color_clasting(img,clr_cst_more)
    # h_eq = hist_eq(img)
    
    # # set target filename
    # filename = '{0}_{1}_{2}-{3}.png'
    # # save the augmentation image
    # cv2.imwrite(os.path.join(DATA_SAVE_PATH,AUGMENTATION_TYPE[0],filename.format(class_name,varian,AUGMENTATION_TYPE[0],'0')),img)
    # cv2.imwrite(os.path.join(DATA_SAVE_PATH,AUGMENTATION_TYPE[1],filename.format(class_name,varian,AUGMENTATION_TYPE[1],'0')),sharp)
    # cv2.imwrite(os.path.join(DATA_SAVE_PATH,AUGMENTATION_TYPE[2],filename.format(class_name,varian,AUGMENTATION_TYPE[2],'0')),gaus_blur)
    # cv2.imwrite(os.path.join(DATA_SAVE_PATH,AUGMENTATION_TYPE[5],filename.format(class_name,varian,AUGMENTATION_TYPE[5],'0')),h_eq)
    # for pict_inx in range(len(brigts)):
    #     cv2.imwrite(os.path.join(DATA_SAVE_PATH,AUGMENTATION_TYPE[3],filename.format(class_name,varian,AUGMENTATION_TYPE[3],str(pict_inx+1))),brigts[pict_inx])
    # for pict_inx in range(len(contrs)):
    #     cv2.imwrite(os.path.join(DATA_SAVE_PATH,AUGMENTATION_TYPE[4],filename.format(class_name,varian,AUGMENTATION_TYPE[4],str(pict_inx+1))),contrs[pict_inx])
    # for pict_inx in range(len(combi)):
    #     cv2.imwrite(os.path.join(DATA_SAVE_PATH,AUGMENTATION_TYPE[6],filename.format(class_name,varian,AUGMENTATION_TYPE[6],str(pict_inx+1))),combi[pict_inx])
    # for pict_inx in range(len(cc)):
    #     cv2.imwrite(os.path.join(DATA_SAVE_PATH,AUGMENTATION_TYPE[7],filename.format(class_name,varian,AUGMENTATION_TYPE[7],str(pict_inx+1))),cc[pict_inx])
    
    # bar for the animation on console
#         bar()