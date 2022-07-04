import cv2 as cv
import imutils
import numpy as np

img = cv.imread('images/coins3.jpg')

cinza = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
                                                                                
blur = cv.medianBlur(cinza, 41)

ret, thresh_otsu = cv.threshold(blur, 0, 400, cv.THRESH_OTSU)

kernel = np.ones((7, 1), np.uint8)
opening = cv.morphologyEx(thresh_otsu, cv.MORPH_OPEN, kernel, iterations=1)
dilated = cv.dilate(opening, kernel, iterations=3)

counter = cv.findContours(
    dilated.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
counter = imutils.grab_contours(counter)
objects = len(counter)

cor = (0, 0, 0)
text = "Contador:" + str(objects)
cv.putText(dilated, text, (600, 25),
           cv.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

images = np.concatenate((cinza, dilated), axis=1)

cv.imshow('Originals -> Dilated(qtd)', images)
cv.waitKey(0)
