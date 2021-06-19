import cv2
import numpy as np
from copy import deepcopy
from itertools import combinations

from .utils import radius, norm

classes = ['circle', 'square', 'triangle']

def refine(preds, contour, img, canvas_fill, color, fillColor, size, shapes, shapeIdx, confidence= 0.6):
    FILLED = False
    if shapes[shapeIdx]['filled']:
        FILLED = True

    if shapes[shapeIdx]['refined']:
        return

    pred_idx = int(np.argmax(preds))
    prediction = classes[pred_idx]
    if preds[pred_idx] < confidence:
        return

    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cX, cY)
    
    (x,y,w,h) = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), thickness= cv2.FILLED)
    if FILLED:
        cv2.rectangle(canvas_fill, (x, y), (x+w, y+h), (0, 0, 0), thickness= cv2.FILLED)

    refine_geometric_shape(prediction, contour, img, center, color, fillColor, size, canvas_fill, FILLED)
    
    img_temp = deepcopy(img)
    imgGray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgGray, 0, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:    
        shapes[shapeIdx]['contour'] = contours[0].tolist()
        shapes[shapeIdx]['refined'] = True
    
def refine_geometric_shape(prediction, contour, img, center, color, fillColor, size, canvas_fill, FILLED):
    if prediction == 'circle':
        canvas_temp = np.zeros_like(img)
        cv2.circle(img, center, radius(contour), color= color, thickness= size)
        cv2.circle(canvas_temp, center, radius(contour), color= color, thickness= size)
        if FILLED:
            imgGray = cv2.cvtColor(canvas_temp, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgGray, 0, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas_fill, [contours[0]], -1, fillColor, -1)
    elif prediction == 'square':
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness= size)
        if FILLED:
            cv2.rectangle(canvas_fill, (x, y), (x+w, y+h), fillColor, thickness= -1)
            
    elif prediction == 'triangle':
        min_dist = 10000
        min_dist_comb = None
        extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
        extRight = tuple(contour[contour[:, :, 0].argmax()][0])
        extTop = tuple(contour[contour[:, :, 1].argmin()][0])
        extBot = tuple(contour[contour[:, :, 1].argmax()][0])
        points = [extLeft, extRight, extTop, extBot]
        for c in combinations(points, 2):
            min_local = norm(*c)
            if min_local < min_dist:
                min_dist = min_local
                min_dist_comb = [*c]
        
        del points[points.index(min_dist_comb[0])]
        canvas_temp = np.zeros_like(img)
        for c in combinations(points, 2):
            cv2.line(img, c[0], c[1], color, size)
            cv2.line(canvas_temp, c[0], c[1], color, size)
        if FILLED:
            imgGray = cv2.cvtColor(canvas_temp, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgGray, 0, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas_fill, [contours[0]], -1, fillColor, -1)

def refine_digits(prediction, contour, img, center, color, size):
    (x,y,w,h) = cv2.boundingRect(contour)
    scale = 1
    fontScale = min(w,h)/(25/scale)
    img = cv2.putText(img, prediction, (x, y+h), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale= fontScale, color= color, thickness= size)
    
    return img