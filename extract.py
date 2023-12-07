import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract(original_image, preprocessed_image):
    #apply average filter (20x20)
    filter_size = 20
    kernel = np.ones((filter_size,filter_size),np.float32)/(filter_size*filter_size)
    blurred = cv2.filter2D(preprocessed_image,-1,kernel)

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
    #temp idi instead of binary
    return idi

def extract2(original_image, preprocessed_image, colored_original_image):
    kernel = np.ones((20, 20), np.float32) / (20 * 20)
    blurred = cv2.filter2D(preprocessed_image, -1, kernel)

    # Get intensity difference image (median filter image - blurred image)
    idi = original_image - blurred

    #threshold 
    upper_threshold = 210
    lower_threshold = 100

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

    # Threshold the result
    #_, thresholded_img = cv2.threshold(idi, 0.03, 255, cv2.THRESH_BINARY)

    # Remove components touching the boundary of the binary image
    #thresholded_img = cv2.copyMakeBorder(idi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(idi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or x == idi.shape[1] or y == 0 or y == idi.shape[0] or x + w >= idi.shape[1] or y + h >= idi.shape[0]: #touche le bord
            cv2.drawContours(idi, [contour], 0, 0, 3)
            print("contour deleted")
        else:
            #print contour to check if exist (debug)
            cv2.drawContours(colored_original_image, [contour], 0, (0, 255, 0), 3)


    # Display the final result
    cv2.imshow('Final Result', idi)
    return idi

def extract_sobel(original_image, preprocessed_img):
    #edge detection
    grad_x = cv2.Sobel(preprocessed_img, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(preprocessed_img, cv2.CV_64F, 0, 1, ksize=1)
    edged = cv2.magnitude(grad_x, grad_y)
    edged = np.uint8(edged)
    
    #edged = cv2.equalizeHist(init_grey_img)
    _, edged = cv2.threshold(edged, 20, 255, cv2.THRESH_BINARY)

    cv2.imshow("Final image", edged)
    cv2.waitKey(0)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #filter contours
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCount = []

    #get the contour that looks like a plate
    for c in cnts:
        cv2.drawContours(original_image, [c], 0, (0, 255, 255), 3)
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if(len(approx) == 4 and perimeter > 200):  #perimeter > 200 allows us to reject remaining small contours
            x,y,w,h = cv2.boundingRect(approx)
            
            if w > h: #width > height
                #a licence plate is 520 x 110 mm which mean a w/h ratio of 4.7
                if(w/h < 8 and w/h > 1.5): #reduces a bit of noise
                    NumberPlateCount = approx

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

                    dst = cv2.warpPerspective(original_image,M,(wd,ht))

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

                    fig, axs = plt.subplots(2, 2)
                    axs[0, 0].set_title("grayscale")
                    axs[1, 0].set_title("gray histo")
                    axs[0, 1].set_title("filtered")
                    axs[1, 1].set_title("filtered histo")
                    

                    crp_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) #grayscale
                    axs[0, 0].imshow(crp_gray, cmap='gray')
                    axs[1, 0].hist(crp_gray.ravel(),50,[0,256]) 
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

                    axs[0, 1].imshow(crp_gray, cmap='gray')
                    axs[1, 1].hist(img.ravel(),256,[0,256]) 
                    fig.tight_layout()
                    #plt.show()
                    break

    return NumberPlateCount


def extract_canny(original_image, preprocessed_img):
    #edged = cv2.Canny(preprocessed_img, 150, 200)
    edged = preprocessed_img
    #cv2.imshow("Final image", edged)
    #cv2.waitKey(0)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #filter contours
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCount = []

    #get the contour that looks like a plate
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        x,y,w,h = cv2.boundingRect(approx)
        cv2.rectangle(original_image,(x,y),(x+w,y+h),(255,0,0),2)
        if(len(approx) < 8 and perimeter > 200):  #perimeter > 200 allows us to reject remaining small contours           
            if w > h: #width > height
                NumberPlateCount = approx

                #transformation matrix
                #we need a perspective matrix as if the car is rotated, the output is not a rectangle but a parallelogram


                # reformat input corners to x,y list
                '''
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

                dst = cv2.warpPerspective(original_image,M,(wd,ht))

                #cv2.imwrite("plate" + ".png", dst)
                plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), cmap='gray')
                plt.show()
                '''
                break
    return NumberPlateCount