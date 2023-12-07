import cv2
import numpy as np
from preprocess import preprocess, preprocess2
from extract import extract, extract_sobel, extract_canny
import matplotlib.pyplot as plt

import os

# https://arxiv.org/ftp/arxiv/papers/1710/1710.10418.pdf

if __name__ == '__main__':
    IMG_DEBUG = False

    directory = 'dataset/valid'
    # iterate over files in that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            #load image
            init_colored_img = cv2.imread(f)
            init_grey_img = cv2.cvtColor(init_colored_img, cv2.COLOR_BGR2GRAY) #grayscale
            init_img = init_grey_img


            #init_grey_img = cv2.equalizeHist(init_grey_img)
            

            #pre-processing
            preprocessed_img = preprocess(init_grey_img)

            plate = extract_sobel(init_colored_img, preprocessed_img)
                    
            if len(plate) == 0:
                pass
                #nothing have been detected as a licence plate
                #TODO : reprocess with another algorithm / technique keep rejected
                preprocessed_img = preprocess2(init_grey_img, init_grey_img)
                cv2.imshow("Final image", preprocessed_img)
                cv2.waitKey(0)
                plate = extract_canny(init_colored_img, preprocessed_img)
                if len(plate) == 0:
                    print("rejected")
                else:                      
                    cv2.drawContours(init_colored_img, [plate], -1, (0, 255, 0), 3)

                cv2.imshow("Final image", init_colored_img)
                cv2.waitKey(0)
            else:                      
                cv2.drawContours(init_colored_img, [plate], -1, (0, 255, 0), 3)
                cv2.imshow("Final image", init_colored_img)
                cv2.waitKey(0)