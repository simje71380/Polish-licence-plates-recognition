import cv2
import matplotlib.pyplot as plt
import numpy as np

def non_max_suppression(rectangles):
    if len(rectangles) == 0:
        return []

    # Initialize the list of selected rectangles
    selected_rectangles = []

    # Browse remaining boxes
    for i, rect in enumerate(rectangles):
        x, y, w, h = rect

        # Check if the rectangle is included in a rectangle
        is_inside = False
        for j in range(len(rectangles)):
            if(i == j):
                continue

            x1, y1, w1, h1 = rectangles[j]

            # Check if the rectangle is completely enclosed in another one
            
            if x1 <= x and x1+w1 >= x + w:
                is_inside = True
                break
            

        # Add rectangle to list of selected rectangles if not included in another one
        if not is_inside:
            selected_rectangles.append(rect)

    return selected_rectangles


def Segmentation(image):
    if image is None:
        print("Rejected: no plate image")
        exit()


    image = cv2.resize(image, (500, 100), interpolation = cv2.INTER_LINEAR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #bilateral filtering
    filtered = cv2.bilateralFilter(gray, 3, 50, 50) 


    #histogram equalization
    hist_equal = cv2.equalizeHist(filtered)

    _, thresholded = cv2.threshold(hist_equal, 75, 255, cv2.THRESH_BINARY)

    #DETECTION VIA SOBEL WITHOUT ADAPTIVE THRESHOLD
    grad_x = cv2.Sobel(thresholded, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(thresholded, cv2.CV_64F, 0, 1, ksize=1)
    edges = cv2.magnitude(grad_x, grad_y)
    edges = np.uint8(edges)

    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # dilatation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Contour detection
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # List for storing rectangles
    rectangles = []

    # Drawing contours
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if(area < 300):
            continue
        #cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)


        x,y,w,h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        if(aspectRatio < 1):
            rectangles.append((x, y, w, h))
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255),3)


    rectangles = sorted(rectangles, key=lambda x: x[2] * x[3], reverse=True)

    # Rectangle filtering
    selected_rectangles = non_max_suppression(rectangles)

    #sorting from lowest x to highest x
    selected_rectangles = sorted(selected_rectangles, key=lambda x: x, reverse=False)

    letters = []
    # Draw selected rectangles on the image
    for rect in selected_rectangles:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        crop_img = image[y:y+h, x:x+w]
        letters.append(crop_img)

    # Show images
    plt.figure(figsize=(12, 12))
    plt.subplot(331), plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)), plt.title('Niveaux de Gris')
    #plt.subplot(222), plt.imshow(cv2.cvtColor(adaptive_thresh, cv2.COLOR_BGR2RGB)), plt.title('Seuil Adaptatif')

    plt.subplot(332), plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)), plt.title('bilateral filtering')
    plt.subplot(333), plt.imshow(cv2.cvtColor(hist_equal, cv2.COLOR_BGR2RGB)), plt.title('histogram equalization')
    plt.subplot(334), plt.imshow(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB)), plt.title('thresholding')
    plt.subplot(335), plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)), plt.title('Sobel Edge Detection')
    plt.subplot(336), plt.imshow(cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)), plt.title('dilated')
    plt.subplot(337), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Résultat Final')
    plt.subplot(338), plt.imshow(cv2.cvtColor(letters[0], cv2.COLOR_BGR2RGB)), plt.title('segment 1')
    plt.subplot(339), plt.imshow(cv2.cvtColor(letters[1], cv2.COLOR_BGR2RGB)), plt.title('segment 2')
    plt.show()

    return letters
