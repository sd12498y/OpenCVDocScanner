import numpy as np 
import cv2
import imutils
from transformation import four_point_transform
from textreg import imgToText
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ocr_simple.py image.jpg')
        sys.exit(1)
    # Read image path from command line
    imPath = sys.argv[1]


    # Read image from disk
    img = cv2.imread(imPath, cv2.IMREAD_COLOR)

    #resize the source img
    shape = img.shape
    img = cv2.resize(img, (int(shape[0]/3),int(shape[1]/3)), interpolation = cv2.INTER_CUBIC)
    #convert it to gray color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 70, 200)
    ret, thresh = cv2.threshold(edged, 127, 255, 0)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)


    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()
    cv2.drawContours(img_copy, contours, -1, (0,255,0), 3)

    warped = four_point_transform(img, contours)

    gray1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (3, 3), 0)
    gaus = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 435, 1)
    
    
    dilated_img = cv2.dilate(warped, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(warped, bg_img)

    #print("before_nor")
    #print(diff_img)
    norm_img = diff_img.copy() # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    #print("after_nor")
    #print(norm_img)
    new_img1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 435, 1)

    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    gray2 = cv2.cvtColor(thr_img, cv2.COLOR_BGR2GRAY)
    gaus2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 435, 1)





    text = imgToText(gaus)
    #print(text)
    text1 = imgToText(gaus2)
    
    print(text1)

    #cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    #cv2.imshow('image', img)
    #cv2.imshow('gray', gray)
    cv2.imshow('blur', diff_img)
    cv2.imshow('edge', bg_img)
    cv2.imshow('warped', warped)
    cv2.imshow('123', gaus2)
    #cv2.imshow('gaus', gaus)
    cv2.waitKey(0)
    cv2.destroyAllWindows



