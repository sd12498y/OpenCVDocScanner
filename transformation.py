# import the necessary packages
import numpy as np
import cv2
 
def order_points(contours):
    rect = np.zeros((4,2),dtype="float32")
    #print(rect)
    array = []
    for points in contours:
        for point in points:
            for item in point:
                array.append([item[0],item[1]])
    array = np.array(array, dtype="float32")
    array_sum = np.sum(array, axis=1)
    array_diff = np.diff(array, axis=1)
    rect[0] = array[np.argmin(array_sum)]
    rect[1] = array[np.argmin(array_diff)]
    rect[2] = array[np.argmax(array_sum)]
    rect[3] = array[np.argmax(array_diff)]
    #print (rect)

    return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped