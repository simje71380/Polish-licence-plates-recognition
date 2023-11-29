#pip freeze > requirements.txt #create requirements.txt (best done in virtual env)
#pip install -r requirements.txt #install requirements.txt (best done in virtual env)

import matplotlib.pyplot as plt


#step 1 : load image and detect plate
img = plt.imread("dataset/train/askasnk_jpg.rf.6676b538895dc0b4e960a7a5f1dcd7a0.jpg")
print(img.shape)
plt.imshow(img)
plt.show() #show the image

#step 2 : detect plates

#step 3 : character extraction

#step 4 : character classification