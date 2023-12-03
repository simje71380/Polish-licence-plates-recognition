import cv2
import numpy as np

def preprocess(img):
    median = cv2.medianBlur(img, 3) #reduce noise
    return median
        
def preprocess2(original_image, img):
    median = cv2.medianBlur(img, 3) #reduce noise
    filter_size = 20
    kernel = np.ones((filter_size,filter_size),np.float32)/(filter_size*filter_size)
    blurred = cv2.filter2D(img,-1,kernel)

    #get intensity difference image (preprocessed_image - blurred image)
    idi = original_image - blurred
    #at this step, the licence plate number are neither black or white but grey
    #so we will remove all dark and white pixels

    
    #threshold 
    upper_threshold = 180
    lower_threshold = 30

    # grab the image dimensions
    h = idi.shape[0]
    w = idi.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            if idi[y, x] > upper_threshold:
                idi[y, x] = 0
            if idi[y, x] < lower_threshold:
                idi[y, x] = 0

    #convert to binary image
    _, binary = cv2.threshold(idi, 1, 255, cv2.THRESH_BINARY)
    return binary