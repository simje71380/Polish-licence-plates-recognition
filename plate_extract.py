import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_plate(img):
    #load image
    init_colored_img = img

    #display setup
    fig, axs = plt.subplots(4, 4)
    

## Preprocessing


    #resize and convert to grayscale
    init_colored_img = cv2.resize(init_colored_img, (640, 640), interpolation = cv2.INTER_LINEAR)
    axs[0, 0].set_title("initial image")
    axs[0, 0].imshow(init_colored_img)

    init_grey_img = cv2.cvtColor(init_colored_img, cv2.COLOR_BGR2GRAY) #grayscale
    axs[1, 0].set_title("grayscale image")
    axs[1, 0].imshow(init_grey_img, cmap='gray')

    #bilateral filtering
    filtered = cv2.bilateralFilter(init_grey_img, 15, 75, 75) 
    axs[2, 0].set_title("bilateral filtering")
    axs[2, 0].imshow(filtered, cmap='gray')

    #histogram equalization
    hist_equal = cv2.equalizeHist(filtered)
    axs[3, 0].set_title("grayscale image equalized")
    axs[3, 0].imshow(hist_equal, cmap='gray')


    #Morphological opening operation using disc shaped structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(200, 200))            
    opening = cv2.morphologyEx(hist_equal, cv2.MORPH_OPEN, kernel)
    axs[0, 1].set_title("opening")
    axs[0, 1].imshow(opening.copy(), cmap='gray')

    hist_equal = hist_equal - opening
    axs[1, 1].set_title("hist_gray - opening")
    axs[1, 1].imshow(hist_equal.copy(), cmap='gray')


    #Binarization via otsu thresholding
    (T, thresh) = cv2.threshold(hist_equal, 0, 255, cv2.THRESH_OTSU)
    print("[INFO] otsu's thresholding value: {}".format(T))
    axs[2, 1].set_title("binary otsu thresholding")
    axs[2, 1].imshow(thresh.copy(), cmap='gray')


## Edge extraction
    
    #via sobel edge operator
    grad_x = cv2.Sobel(thresh, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(thresh, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)
    edges = np.uint8(edges)
    axs[3, 1].set_title("edge detection sobel")
    axs[3, 1].imshow(edges, cmap='gray')


    #dilatation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    axs[0, 2].set_title("dilatation")
    axs[0, 2].imshow(dilated, cmap='gray')


    #filling holes
    contour,_ = cv2.findContours(dilated,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    cpy = dilated.copy()

    for cnt in contour:
        cv2.drawContours(image=cpy, contours=[cnt], contourIdx=-1, color=(255,255,255), thickness=cv2.FILLED)

    axs[1, 2].set_title("filling holes")
    axs[1, 2].imshow(cpy, cmap='gray')

    #Morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))    
    opening = cv2.morphologyEx(cpy, cv2.MORPH_OPEN, kernel, iterations = 5)

    axs[2, 2].set_title("filled opened")
    axs[2, 2].imshow(opening.copy(), cmap='gray')


## Contour detection and analysis

    cnts, _ = cv2.findContours(opening.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #filter contours
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]

    all_contour_image = init_colored_img.copy()

    candidates = []

    for c in cnts:
        rect = cv2.minAreaRect(c) 
        box = cv2.boxPoints(rect) 
        box = np.int0(box)
        perimeter = cv2.arcLength(box, True)
        area = cv2.contourArea(box)
        if(perimeter < 250 or area < 2000): #allows us to reject remaining small contours
            continue

        #or perimeter/area < 0.07

        #print("perimeter :" + str(perimeter))
        #print("area :" + str(area))
        #print("area/perimeter : " + str(area/perimeter))
        img = cv2.drawContours(all_contour_image, [box], 0, (0, 255, 0), 2)
        candidates.append(box)
    
    axs[3, 2].set_title("detected candidates")
    axs[3, 2].imshow(all_contour_image)

    if len(candidates) == 0:
        print("REJECTED : no plate found")
        return []
    else:
        #TODO : classification of candidate via texture analysis ?
        pass

    #display of the detected plate
    axs[0, 3].set_title("detected plate")
    #axs[0, 3].imshow(plate)

    

                    
    plt.show() #display

    #save image to disk
