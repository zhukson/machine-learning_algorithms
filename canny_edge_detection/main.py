import numpy as np
from PIL import Image
from canny_edge_detection import *

IMAGE_PATH = "lenna.png"

if __name__ == "__main__":
    c1 = canny_detection(IMAGE_PATH)
    c1.detect()
