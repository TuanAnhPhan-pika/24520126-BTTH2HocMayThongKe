import numpy as np 
import idx2numpy
from matplotlib import pyplot as plt 
import cv2 as cv
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    images = idx2numpy.convert_from_file(str(data_dir / "train-images.idx3-ubyte"))
    labels = idx2numpy.convert_from_file(str(data_dir / "train-labels.idx1-ubyte"))

    image = images[100]
    label = labels[100]

    cv.imshow("Show digital number", image)
    print("Label of the image is:", label)
    cv.waitKey(0)
    cv.destroyAllWindows()
