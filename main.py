from plate_extract import extract_plate
from segmentation import Segmentation
from OCRExploit import predict
import cv2, os


if __name__ == "__main__":
    directory = 'dataset/new2'
    # iterate over files in that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            init_colored_img = cv2.imread(f)
            plate = extract_plate(init_colored_img)

            plate = cv2.imread("Plaque.jpg")
            if(len(plate) != 0):
                letters = Segmentation(plate)
                if(len(letters) > 0):
                    predict(letters)