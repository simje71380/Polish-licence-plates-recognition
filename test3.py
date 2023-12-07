import cv2
import numpy as np
from preprocess import preprocess, preprocess2
from extract import extract, extract_sobel, extract_canny
import matplotlib.pyplot as plt

import os

# https://arxiv.org/ftp/arxiv/papers/1710/1710.10418.pdf

def are_values_equal_with_tolerance(value1, value2, tolerance_percent=10):
    tolerance = (tolerance_percent / 100) * max(abs(value1), abs(value2))
    return abs(value1 - value2) <= tolerance


if __name__ == '__main__':
    directory = 'dataset/valid'
    # iterate over files in that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            #load image
            init_colored_img = cv2.imread(f)
            init_grey_img = cv2.cvtColor(init_colored_img, cv2.COLOR_BGR2GRAY) #grayscale


            #display setup
            fig, axs = plt.subplots(3, 3)
            axs[0, 0].set_title("initial image")
            axs[0, 0].imshow(init_colored_img)

            axs[1, 0].set_title("grayscale image")
            axs[1, 0].imshow(init_grey_img, cmap='gray')


            img = cv2.equalizeHist(init_grey_img)
            axs[2, 0].set_title("grayscale image equalized")
            axs[2, 0].imshow(img, cmap='gray')

            img = init_grey_img # DO NOT EQUALIZE


            #filtrage
                # find frequency of pixels in range 0-255 
            #plt.figure()
            #plt.hist(img.ravel(),256,[0,256])
            
            blured = preprocess(img)
            axs[0, 1].set_title("blured (preprocess)")
            axs[0, 1].imshow(blured, cmap='gray')

            binary = extract(init_grey_img, blured)
            axs[1, 1].set_title("binary idi")
            axs[1, 1].imshow(binary.copy(), cmap='gray')



            #step 2 : detect plates
                #step 2.2 détection des bords

            #application d'un filtre Sobel
            grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(grad_x, grad_y)
            edges = np.uint8(edges)
            axs[2, 1].set_title("edge detection (not used)")
            axs[2, 1].imshow(edges, cmap='gray')

            #threshold pour avoir une image binaire
            #_, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)


            cnts, _ = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            #filter contours
            cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]

            all_contour_image = init_colored_img.copy()

            zone_of_interest = []
            plate_found = False

            for c in cnts:
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
                area = cv2.contourArea(c)
                if(perimeter < 70 or area < 1000): #allows us to reject remaining small contours
                    continue

                if(len(approx) != 4):
                    x,y,w,h = cv2.boundingRect(approx)
                    if(w/h > 8 or w<h):
                        continue
                    cv2.drawContours(all_contour_image, [c], 0, (0, 255, 255), 3)

                    #in this case, there might be some interesting areas.
                    #TODO: create the bounding box
                    #      implement another system : sobel/canny edge detection on fresh image only in the area we want
                    #      try to detect a plate. If for no plate is detected, reject the image
                    zone_of_interest.append([x, y, w, h])

                if(len(approx) == 4):
                    print(cv2.contourArea(c))
                    x,y,w,h = cv2.boundingRect(approx)
                    if(w/h > 8):
                        continue  #w/h < 8 reject lines


                    #transformation matrix
                    #we need a perspective matrix as if the car is rotated, the output is not a rectangle but a parallelogram

                    # reformat input corners to x,y list
                    sortcorners = []
                    for corner in approx:
                        pt = [corner[0][0], corner[0][1]]
                        sortcorners.append(pt)

                    # Sort corners based on y-coordinate
                    sortcorners.sort(key=lambda elem: elem[1])

                    output = sortcorners.copy()
                    if(sortcorners[0][0] < sortcorners[1][0]):
                        output[3] = sortcorners[1]
                        output[0] = sortcorners[0]
                    else:
                        output[3] = sortcorners[0]
                        output[0] = sortcorners[1]

                    if(sortcorners[2][0] < sortcorners[3][0]):
                        output[1] = sortcorners[2]
                        output[2] = sortcorners[3]
                    else:
                        output[1] = sortcorners[3]
                        output[2] = sortcorners[2]

                    icorners = np.float32(output)

                    # Get corresponding output corners from width and height
                    wd, ht = 500, 300
                    ocorners = np.float32([[0, 0], [0, ht], [wd, ht], [wd, 0]])

                    # Get perspective transformation matrix
                    M = cv2.getPerspectiveTransform(icorners, ocorners)

                    dst = cv2.warpPerspective(init_colored_img,M,(wd,ht))

                    #TODO : ajouter une étape de vérification : exemple : texture analysis

                    axs[2, 2].set_title("extracted plate")
                    axs[2, 2].imshow(dst)

                    #cv2.imwrite("plate" + ".png", dst)


                    #TODO: as we said we have to detect only black and white plates
                    #so we should see a pick in low values for black (<75) and at high value (>180) white
                    #if not the case we should reject it


                    #thresholds 
                    upper_threshold = 180
                    lower_threshold = 50

                    # grab the image dimensions
                    h = dst.shape[0]
                    w = dst.shape[1]
                    

                    crp_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) #grayscale

                    crp_gray = cv2.equalizeHist(crp_gray)
                    img = crp_gray.copy()

                    # loop over the image, pixel by pixel
                    for y in range(0, h):
                        for x in range(0, w):
                            if crp_gray[y, x] > upper_threshold:
                                img[y, x] = 220
                            elif crp_gray[y, x] < lower_threshold:
                                img[y, x] = 25
                            else:
                                img[y, x] = 120

                    cv2.drawContours(all_contour_image, [c], 0, (255, 0, 255), 3) #purple = detected as a plate
                    plate_found = True
                    break

            axs[2, 1].set_title("all contours")
            axs[2, 1].imshow(all_contour_image)

            if(not plate_found):
                #explore zones of interest
                for x,y,w,h in zone_of_interest[::1]:
                    fig2, axs2 = plt.subplots(2,2)

                    cv2.rectangle(all_contour_image, (x, y), (x + w, y + h), (255,0,0), 4)

                    x_margin = int(0.2*w) #adding 20% width each side
                    y_margin = int(0.2*h) #adding 20% height each side

                    y0 = y-y_margin
                    if(y0 < 0): y0 = 0

                    x0 = x-x_margin
                    if(x0 < 0): x0 = 0

                    y1 = y+h+y_margin
                    if(y1 > np.shape(init_colored_img)[0]): y1 = np.shape(init_colored_img)[0]

                    x1 = x+w+x_margin
                    if(x1 > np.shape(init_colored_img)[1]): x1 = np.shape(init_colored_img)[1]

                    crop_img = init_colored_img[y0:y1, x0:x1]
                    if(np.shape(crop_img)[0] == 0 or np.shape(crop_img)[1] == 0 ):
                        continue


                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) #grayscale  

                    axs2[0, 0].set_title("zone of interest")
                    axs2[0, 0].imshow(crop_img)


                    preprocessed = cv2.medianBlur(gray, 3) #reduce noise
                    
                    axs2[0, 1].set_title("preprocessed")
                    axs2[0, 1].imshow(preprocessed, cmap="gray")

                    #sobel edge detection
                    #gray_binary = cv2.threshold(preprocessed, 1, 255, cv2.THRESH_BINARY)
                    grad_x = cv2.Sobel(preprocessed.copy(), cv2.CV_64F, 1, 0, ksize=1)
                    grad_y = cv2.Sobel(preprocessed.copy(), cv2.CV_64F, 0, 1, ksize=1)
                    edges = cv2.magnitude(grad_x, grad_y)
                    edges = np.uint8(edges)

                    _, binary_edges = cv2.threshold(edges, 25, 255, cv2.THRESH_BINARY) #20 is good

                    #binary_edges = cv2.Canny(preprocessed.copy(), 1, 255)

                    axs2[1, 0].set_title("edge binary")
                    axs2[1, 0].imshow(binary_edges, cmap="gray")


                    cnts, _ = cv2.findContours(binary_edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                    #filter contours
                    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]

                    contour_img = crop_img.copy()

                    for c in cnts:
                        perimeter = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.05 * perimeter, True)
                        area = cv2.contourArea(c)
                        #if(perimeter < 70 or area < 1000): #allows us to reject remaining small contours
                        #    continue

                        if(area < 1300):
                            continue

                        cv2.drawContours(contour_img, [c], 0, (0, 255, 255), 3)

                        if(len(approx) == 4):
                            x,y,w,h = cv2.boundingRect(approx)
                            if(w/h > 8 and w/h < 1.4):
                                continue  #w/h < 8 reject lines
                           
                            print("contour area : " + str(area))
                            print("boudning box area : " + str(cv2.contourArea(approx)))

                            #compare bounding box area and contour are -> should be pretty much the same
                            if(not are_values_equal_with_tolerance(area, cv2.contourArea(approx), 20)):
                                continue #both values are not the same at +-20% -> reject

                            #transformation matrix
                            #we need a perspective matrix as if the car is rotated, the output is not a rectangle but a parallelogram

                            # reformat input corners to x,y list
                            sortcorners = []
                            for corner in approx:
                                pt = [corner[0][0], corner[0][1]]
                                sortcorners.append(pt)

                            # Sort corners based on y-coordinate
                            sortcorners.sort(key=lambda elem: elem[1])

                            output = sortcorners.copy()
                            if(sortcorners[0][0] < sortcorners[1][0]):
                                output[3] = sortcorners[1]
                                output[0] = sortcorners[0]
                            else:
                                output[3] = sortcorners[0]
                                output[0] = sortcorners[1]

                            if(sortcorners[2][0] < sortcorners[3][0]):
                                output[1] = sortcorners[2]
                                output[2] = sortcorners[3]
                            else:
                                output[1] = sortcorners[3]
                                output[2] = sortcorners[2]

                            icorners = np.float32(output)

                            # Get corresponding output corners from width and height
                            wd, ht = 500, 300
                            ocorners = np.float32([[0, 0], [0, ht], [wd, ht], [wd, 0]])

                            # Get perspective transformation matrix
                            M = cv2.getPerspectiveTransform(icorners, ocorners)

                            dst = cv2.warpPerspective(crop_img,M,(wd,ht))

                            axs[2, 2].set_title("extracted plate")
                            axs[2, 2].imshow(dst)

                            #cv2.imwrite("plate" + ".png", dst)
                            cv2.drawContours(contour_img, [c], 0, (255, 0, 0), 3)
                            plate_found = True
                            break

                    axs2[1, 1].set_title("contours")
                    axs2[1, 1].imshow(contour_img)
                    if(plate_found):
                        break

            axs[0, 2].set_title("zones of interest")
            axs[0, 2].imshow(all_contour_image)
                            
            plt.show() #display