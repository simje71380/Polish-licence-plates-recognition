import cv2
import numpy as np
from preprocess import preprocess
from extract import extract, extract2
import matplotlib.pyplot as plt

import os

# https://arxiv.org/ftp/arxiv/papers/1710/1710.10418.pdf

if __name__ == '__main__':
    IMG_DEBUG = True

    directory = 'dataset/train'
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

            #Extraction of licence plate region
            binary = extract(init_grey_img, preprocessed_img)

            #extract2(init_img, preprocessed_img, init_colored_img)

            # Sobel edge filter
            grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(grad_x, grad_y)
            edges = np.uint8(edges)


            #look for rectangles
            cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                peri = cv2.arcLength(c, True)
                if peri > 200 :
                    #cv2.drawContours(init_colored_img, c, -1, (0,255,255), 3)
                    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
                    x,y,w,h = cv2.boundingRect(approx)
                    #a licence plate always a higher width than a height so reject if h > w
                    if w > h:
                        #a licence plate is 520 x 110 mm which mean a w/h ratio of 4.7
                        if(w/h < 8 and w/h > 1.3): #reduces a bit of noise
                            cv2.rectangle(init_colored_img,(x,y),(x+w,y+h),(255,0,0),2)
                
            plt.figure()
            plt.imshow(init_colored_img)
            plt.show()

            
            #display
            if(IMG_DEBUG):
                #downsize images to fit in screen
                scale_percent = 50 # percent of original size
                width = int(init_grey_img.shape[1] * scale_percent / 100)
                height = int(init_grey_img.shape[0] * scale_percent / 100)
                dim = (width, height)
                
                # resize images
                resized_init = cv2.resize(init_grey_img, dim, interpolation = cv2.INTER_AREA)
                resized_img = cv2.resize(preprocessed_img, dim, interpolation = cv2.INTER_AREA)
                resized_preprocessed_img = cv2.resize(binary, dim, interpolation = cv2.INTER_AREA)
                resized_idi = cv2.resize(edges, dim, interpolation = cv2.INTER_AREA)

                compare = np.concatenate((resized_init, resized_img), axis=1) #side by side comparison
                compare2 = np.concatenate((resized_preprocessed_img, resized_idi), axis=1)
                compare = np.concatenate((compare, compare2), axis=0)
                cv2.imshow('noise reduction (Pre-processing step)', compare)
                cv2.waitKey(0)
                cv2.destroyAllWindows
            