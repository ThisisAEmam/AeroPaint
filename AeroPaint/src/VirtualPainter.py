'''
Virtual Painter module.
'''

import time
from copy import deepcopy
from cv2 import cv2
import numpy as np
from .model_predictions import predict
from .HandTrackingModule import HandDetector
from .utils import single_true, is_in_range, contourIntersect, img_with_canvas
from .refine import refine
import matplotlib.pyplot as plt

def virtualPainter(e, update, cap, videoStop):   
    headers = {
            'main-header': './assets/main-header.png',
            'brush-default': './assets/brush-default.png',
            'brush-red': './assets/brush-red.png',
            'brush-green': './assets/brush-green.png',
            'brush-blue': './assets/brush-blue.png',
            'fill-default': './assets/fill-default.png',
            'fill-red': './assets/fill-red.png',
            'fill-green': './assets/fill-green.png',
            'fill-blue': './assets/fill-blue.png',
            'eraser-10': './assets/eraser-10.png',
            'eraser-20': './assets/eraser-20.png',
            'eraser-40': './assets/eraser-40.png',
        }
    
    headerImg = cv2.imread(headers['main-header'])
    detector = HandDetector(maxHands=1, detectionCon=0.9)
    canvas = np.zeros((720,1280,3),np.uint8)
    canvas_temp = np.zeros_like(canvas)
    canvas_fill = np.zeros_like(canvas)
    canvasErase = np.zeros_like(canvas)
    xp,yp = 0,0
    t1 = time.time()
    
    opType = 'draw'
    colors = {
            'default': None,
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
        }
    
    drawColor = colors['blue']
    fillColor = colors['blue']
    brushSize = 12
    eraserSize = 20
    
    ERASING = False
    REFINE = False
    
    shapes = []
    erased_shapes = []
  
    while True:
        suc,img=cap.read()
        img = cv2.resize(img, (1280, 720))
        img=cv2.flip(img,1)
        img=detector.findHands(img)
        img[:125, :] = headerImg
        landmarks,bbox=detector.findPosition(img,draw=False)
        if len(landmarks)!=0:
            x_index, y_index=landmarks[8][1:]
            x_middle,y_middle=landmarks[12][1:]
            fingers=detector.fingersUp()
            cv2.circle(img, (int(x_index),int(y_index)), 5,(0,0,0), cv2.FILLED)
            if y_index <= 125:
                if is_in_range((855, 62), (x_index, y_index)):
                    opType = 'draw'
                    headerImg = cv2.imread(headers['brush-default'])
                    drawColor = colors['default']
                elif is_in_range((680, 62), (x_index, y_index)) and opType == 'draw':
                    headerImg = cv2.imread(headers['brush-blue'])
                    drawColor=colors['blue']
                elif is_in_range((580, 62), (x_index, y_index)) and opType == 'draw':
                    headerImg = cv2.imread(headers['brush-green'])
                    drawColor=colors['green']
                elif is_in_range((480, 62), (x_index, y_index)) and opType == 'draw':
                    headerImg = cv2.imread(headers['brush-red'])
                    drawColor=colors['red']
                elif is_in_range((1010, 62), (x_index, y_index)):
                    opType = 'erase'
                    eraserSize = 20
                    headerImg = cv2.imread(headers['eraser-20'])
                elif is_in_range((680, 62), (x_index, y_index)) and opType == 'erase':
                    eraserSize = 40
                    headerImg = cv2.imread(headers['eraser-40'])
                elif is_in_range((580, 62), (x_index, y_index)) and opType == 'erase':
                    eraserSize = 20
                    headerImg = cv2.imread(headers['eraser-20'])
                elif is_in_range((480, 62), (x_index, y_index)) and opType == 'erase':
                    eraserSize = 10
                    headerImg = cv2.imread(headers['eraser-10'])
                elif is_in_range((1180, 62), (x_index, y_index)):
                    opType = 'fill'
                    headerImg = cv2.imread(headers['fill-default'])
                elif is_in_range((680, 62), (x_index, y_index)) and opType == 'fill':
                    headerImg = cv2.imread(headers['fill-blue'])
                    fillColor=colors['blue']
                elif is_in_range((580, 62), (x_index, y_index)) and opType == 'fill':
                    headerImg = cv2.imread(headers['fill-green'])
                    fillColor=colors['green']
                elif is_in_range((480, 62), (x_index, y_index)) and opType == 'fill':
                    headerImg = cv2.imread(headers['fill-red'])
                    fillColor=colors['red']
    
            if fingers[1] and single_true(fingers):
                t1=time.time()
                if opType == 'draw' and y_index > 125 and drawColor:
                    cv2.circle(img,(x_index,y_index), brushSize, drawColor,cv2.FILLED)
                    if xp==0 and yp==0:
                        xp,yp=x_index,y_index
                    cv2.line(img,(xp,yp),(x_index,y_index), drawColor, brushSize)
                    cv2.line(canvas, (xp, yp), (x_index, y_index), drawColor, brushSize)
                    cv2.line(canvas_temp, (xp, yp), (x_index, y_index), drawColor, brushSize)
                    xp,yp=x_index,y_index
    
                elif opType == 'fill' and y_index > 125 and fillColor:
                    cv2.circle(img,(x_index,y_index), brushSize, fillColor, cv2.FILLED)
                    filled_contours = []
                    for shape in shapes:
                        contour = np.array(shape['contour'])
                        is_point_inside_contour = cv2.pointPolygonTest(contour, (x_index,y_index), False)
                        distance_to_contour = cv2.pointPolygonTest(contour, (x_index,y_index), True)
                        if is_point_inside_contour >= 0:
                            filled_contours.append((shape, distance_to_contour))
    
                    if len(filled_contours) != 0:
                        min_dist = 1000000
                        deepest_contour = None
                        deepest_shape = None
                        for cont in filled_contours:
                            if cont[1] < min_dist:
                                min_dist = cont[1]
                                deepest_contour = np.array(cont[0]['contour'])
                                deepest_shape = cont[0]
    
                        (x,y,w,h) = cv2.boundingRect(deepest_contour)
                        cv2.drawContours(canvas_fill, [deepest_contour], -1, fillColor, -1)
                        canvas_fill[y:y+h, x:x+w] = canvas_fill[y:y+h, x:x+w] - canvas[y:y+h, x:x+w]
                        shapes[shapes.index(deepest_shape)]['filled'] = True
    
                        for shape in shapes:
                            if not shape['filled']:
                                contour = np.array(shape['contour'])
                                cv2.drawContours(canvas_fill, [contour], -1, (0, 0, 0), -1)
    
                elif opType == 'erase':
                    ERASING = True
                    xp, yp = 0, 0
                    cv2.circle(img,(int(x_index),int(y_index)), eraserSize,(255,255,255), cv2.FILLED)
                    cv2.circle(canvas,(int(x_index),int(y_index)), eraserSize,(0,0,0), cv2.FILLED)
                    cv2.circle(canvas_fill,(int(x_index),int(y_index)), eraserSize,(0,0,0), cv2.FILLED)
                    cv2.circle(canvasErase,(int(x_index),int(y_index)), eraserSize,(255,255,255), 2)
                    eraserCircle = None
                    imgGray = cv2.cvtColor(canvasErase, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgGray, 0, 255, 0)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cont in contours:
                        eraserCircle = cont
                    for shape in shapes:
                        canvasErase = cv2.drawContours(canvasErase, [np.array(shape['contour'])], 0, drawColor, 1)
                    for i, shape in enumerate(shapes):
                        if contourIntersect(canvasErase, eraserCircle, np.array(shape['contour'])):
                            if not i in erased_shapes:
                                erased_shapes.append(i)
                    canvasErase = np.zeros_like(canvas)
    
            else:
                if int(time.time()-t1) > 0.2 and (opType == 'draw' or opType == 'fill'):
                    detector.fillPoly(canvas_temp, drawColor)
                    imgGray = cv2.cvtColor(canvas_temp, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgGray, 0, 255, 0)
                    ret, prediction_shape = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 0, 255,cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cont in contours:
                        contour = cont.tolist()
                        shapes_contours = list(map(lambda x: x['contour'], shapes))
                        if not contour in shapes_contours:
                            shapes.append({'contour': contour, 'filled': False, 'refined': False})
                    canvas_temp = np.zeros_like(canvas)
                    if detector.isNotRefine():
                        REFINE = False
                    
                    if detector.isRefine() and not REFINE:
                        for i, shape in enumerate(shapes):
                            if not shape['refined']:
                                cont = np.array(shape['contour'])
                                (x,y,w,h) = cv2.boundingRect(cont)
                                preds = predict(cv2.cvtColor(prediction_shape[y-10:y+h+10, x-10:x+w+10], cv2.COLOR_GRAY2RGB))
                                refine(preds, cont, canvas, canvas_fill, drawColor, fillColor, brushSize, shapes, i, confidence=1)
                        REFINE = True
    
                elif int(time.time()-t1) > 0.2 and opType == 'erase' and ERASING:
                    shapes_temp = deepcopy(shapes)
                    for i in erased_shapes:
                        filled = shapes_temp[i]['filled']
                        refined = shapes_temp[i]['refined']
                        del shapes_temp[i]
                        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgGray, 0, 255, 0)
                        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        for cont in contours:
                            contour = cont.tolist()
                            shapes_contours = list(map(lambda x: x['contour'], shapes))
                            if len(contour) < 10:
                                continue
                            if not contour in shapes_contours:
                                shapes_temp.append({'contour': contour, 'filled': filled, 'refined': refined})
                    shapes = shapes_temp
                    ERASING = False
    
                xp, yp = 0, 0

        ######################## KERNELS AND IMAGE PROCESSING OPERATIONS #########################
        img = img_with_canvas(img, canvas, canvas_fill)
        e.processEvents()
        blank = np.ones_like(canvas)*255
        blank = img_with_canvas(blank, canvas, canvas_fill)
        update(img, blank)
        
        if videoStop():
            cap.release()
            break
