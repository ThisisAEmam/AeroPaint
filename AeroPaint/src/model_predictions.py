import cv2
import matplotlib.pyplot as plt
import numpy as np
from .model_arch import create_model


model = create_model(3)
model.load_weights('../model/weights')

def squaring_img(img):
    w, h = img.shape[:2]
    if w == h:
        return img
    maxVal = max(w, h)
    diff = np.abs(w - h)
    newImg = np.zeros((maxVal, maxVal, 3), dtype=np.uint8)
    if w > h:
        newImg[:, int(diff/2):int(diff/2)+h] = img
    else:
        newImg[int(diff/2):int(diff/2)+w, :] = img
    return newImg
        

def predict(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img = cv2.erode(img, kernel, iterations=1)
    img = squaring_img(img)
    img_resize = (cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC))/255.
    preds = model(img_resize[np.newaxis,...]).mean().numpy()[0]
    return preds
