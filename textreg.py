import numpy as np 
import cv2
import sys
import pytesseract
import os
from PIL import Image


def imgToText(img):
    config = ('-l eng --oem 1 --psm 3')
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, img)
    text = pytesseract.image_to_string(Image.open(filename), config=config)
    os.remove(filename)
    #print(text)
    return text
    