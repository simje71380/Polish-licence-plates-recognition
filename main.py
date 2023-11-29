#pip freeze > requirements.txt #create requirements.txt (best done in virtual env)
#pip install -r requirements.txt #install requirements.txt (best done in virtual env)

import matplotlib.pyplot as plt

import cv2 


#step 1 : load image

#loading via matplotlib
img = plt.imread("dataset/train/askasnk_jpg.rf.6676b538895dc0b4e960a7a5f1dcd7a0.jpg")

#loading image via cv2
img = cv2.imread("dataset/train/polen17_jpg.rf.67a03dbe77abba286c79c27ea0e9982f.jpg",0) #2nd arg 0 -> grayscale
print(img.shape)
plt.figure()
plt.imshow(img)


#step 2 : detect plates

# find frequency of pixels in range 0-255 
plt.figure()
plt.hist(img.ravel(),256,[0,256]) 


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
plt.imshow(img)
plt.show() #display

#step 3 : character extraction

#step 4 : character classification