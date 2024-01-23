from plate_extract import extract_plate
from segmentation import Segmentation
from OCRExploit import predict
import cv2, os


if __name__ == "__main__":
    directory = 'dataset/new'
    # iterate over files in that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            init_colored_img = cv2.imread(f)
            plates = extract_plate(init_colored_img)
            for p in plates:
                letters = Segmentation(p)
                if(letters is not None):
                    predict(letters)