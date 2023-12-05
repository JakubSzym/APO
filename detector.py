#!/usr/bin/env python3

import cv2 
import numpy as np 
from matplotlib import pyplot as plt

# TODO
# 
# zastosować wszystkie metody dopasowania i zsumować
# usunąć nadmiarowe ramki
#


def performAugmentation(in_image):
    images = []
    images.append(in_image)

    vertical_flip =                cv2.flip(in_image, 0)
    horizontal_flip =              cv2.flip(in_image, 1)
    vertical_and_horizontal_flip = cv2.flip(in_image, -1)

    images.append(vertical_flip)
    images.append(horizontal_flip)
    images.append(vertical_and_horizontal_flip)

    return images

img_rgb = cv2.imread('board.jpg')
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('car.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]

templates = performAugmentation(template)

locations = []

#
# metody dopasowania: TM_SQDIFF_NORMED, TM_CCOEFF_NORMED, TM_CCORR_NORMED (nieunormowane są zbyt czułe)
# TM_SQDIFF_NORMED - najlepsze wyniki dla ufności 0.8 - 0.85
# TM_CCOEFF_NORMED - najlepsze wyniki dla 0.4
# TM_CCORR_NORMED - najlepsze wyniki dla 0.835
#

for t in templates:
    res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where( res >= threshold)
    locations.append(loc)

for loc in locations:
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 3) 

cv2.imwrite('items-found.jpg',img_rgb)