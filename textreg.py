import numpy as np 
import cv2
import sys
import pytesseract
import os
from PIL import Image


def imgToText(img):
    #print("here")
    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    config = ('-l eng --oem 1 --psm 3')
    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, img)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename), config=config)
    os.remove(filename)
    #print(text)
    return text
    