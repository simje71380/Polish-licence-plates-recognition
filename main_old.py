#pip freeze > requirements.txt #create requirements.txt (best done in virtual env)
#pip install -r requirements.txt #install requirements.txt (best done in virtual env)

import matplotlib.pyplot as plt
import numpy as np
import cv2 


# apres detection des plaques : transformation perspective
#https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143


if __name__ == "__main__":
    #step 1 : load image

    #loading via matplotlib
    init_img = plt.imread("dataset/train/askasnk_jpg.rf.6676b538895dc0b4e960a7a5f1dcd7a0.jpg")

    #loading image via cv2
    init_img = cv2.imread("dataset/valid/nazwy_jpg.rf.c268a5b45fcb5f346def9a6b84387791.jpg",0) #2nd arg 0 -> grayscale
    #init_img = cv2.imread('dataset/train/thumb-article-1248-tmain_jpg.rf.0ce02fef5747ab99a8146c1f6eb86829.jpg',0)
    #init_img = cv2.imread('dataset/train/t4_jpg.rf.224f07aab24f0ac28ffdd3b61ef38400.jpg',0)
    
    
    #init_img = cv2.imread('dataset/train/thumb-article-1248-tmain_jpg.rf.0ce02fef5747ab99a8146c1f6eb86829.jpg')
    #init_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grayscale
    #init_img = cv2.GaussianBlur(img, (5, 5), 0) #application d'un filtre gaussien pour réduire le bruit

    img = init_img

    #plt.figure()
    #plt.imshow(img, cmap='gray')

    #plt.figure()
    img = cv2.equalizeHist(img)
    #plt.imshow(img, cmap='gray')


    #filtrage
        # find frequency of pixels in range 0-255 
    #plt.figure()
    #plt.hist(img.ravel(),256,[0,256]) 


    #threshold
    threshold = 200

    # grab the image dimensions
    h = img.shape[0]
    w = img.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            if img[y, x] < threshold:
                img[y, x] = 0
            
    #dans le cas de cette image dataset/train/polen17_jpg.rf.67a03dbe77abba286c79c27ea0e9982f.jpg,
    #on peut facilement trouver la plaque car le blanc est casi uniquement sur la plaque
    plt.figure()
    plt.imshow(img, cmap='gray')


    #step 2 : detect plates
        #step 2.2 détection des bords

    #application d'un filtre Sobel
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)
    edges = np.uint8(edges)
    plt.figure()
    plt.imshow(edges, cmap='gray')

    #threshold pour avoir une image binaire
    _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)


    #Hough Transform pour trouver les lignes
    lines_v = cv2.HoughLines(binary ,rho = 1, theta = np.pi, threshold = 130) #vertical lines detection
    lines_h = cv2.HoughLines(binary, 1, np.pi / 180, threshold=300) #horizontal lines detection
    
    lines = []
    for line in lines_v:
        lines.append(line)
        
    for line in lines_h:
        lines.append(line)
    #process des lignes trouvées

    if lines is not None:
        filtered_lines = []
        for line in lines:
            rho, theta = line[0]
            if (theta > np.pi / 4 and theta < 3 * np.pi / 4) or (theta > 5 * np.pi / 4 and theta < 7 * np.pi / 4): # +/- lignes horizontale
                filtered_lines.append(line)
            elif (theta > -np.pi / 4 and theta < np.pi / 4) or (theta > -5 * np.pi / 4 and theta < 5 * np.pi / 4):  # +/- lignes verticales
                filtered_lines.append(line)

        #affichage des lignes trouvées
        for line in filtered_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(init_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    plt.figure()
    plt.imshow(init_img, cmap='gray')
    plt.show() #display

    #step 3 : character extraction

    #step 4 : character classification