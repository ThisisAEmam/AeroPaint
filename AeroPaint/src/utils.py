import cv2
import numpy as np

def radius(contour):
    return int(np.sqrt(cv2.contourArea(contour)/np.pi))

def norm(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def height_to_scale(height):
    return height * (3 / 4)

def single_true(iterable):
    i = iter(iterable)
    return any(i) and not any(i)

def is_in_range(point, index_point, r=50):
    if index_point[0] in range(point[0] - r, point[0] + r) and index_point[1] in range(point[1] - r, point[1] + r):
        return True
    
def img_with_canvas(img, canvas, canvas_fill):
    imgGray = cv2.cvtColor(canvas_fill, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas_fill)
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)
    return img

def contourIntersect(original_image, contour1, contour2):
    contours = [contour1, contour2]
    blank = np.zeros_like(original_image)

    image1 = cv2.drawContours(blank.copy(), contours, 0, (255, 255, 255), 1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, (255, 255, 255), 1)

    intersection = np.logical_and(image1, image2)

    return intersection.any()
