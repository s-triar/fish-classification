from skimage.feature import hog
import cv2
import time
import glob
import csv
from alive_progress import alive_bar

# Declare path to dataset
DATASET_PATH='dataset/archive/Fish_Data/images/cropped'

# Declare path to target folder to save the feature.
# Create folder manually
DATA_HOG_PATH = 'dataset/fish_data_hog'

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
        # set start time process
        start_time   = time.time()    
        # get hog feature by calling library
        fd = hog(img, orientations=9, pixels_per_cell=(12, 12),
                    cells_per_block=(1, 1), visualize=False)
        # calculate the time it took to process
        elapsed_time = time.time() - start_time
        # write the data into a file    
        f = open(DATA_HOG_PATH+'/'+'hog_fish.csv', 'a+', newline='', encoding='utf-8')
        writer = csv.writer(f)
        h = list(fd)
        h.insert(0,elapsed_time)
        h.insert(0,varian)
        h.insert(0,class_name)
        h.insert(0,idx)
        # write the data according columns below
        # id, class, varian, time, ...hog fts
        writer.writerow(h)
        f.close()
        bar()
