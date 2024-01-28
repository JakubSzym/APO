#!/usr/bin/env python3
from logging import root
import os
import cv2 
import numpy as np 
from matplotlib import pyplot as plt

def performAugmentation(in_image):
    images = []
    images.append(in_image)

    vertical_flip                = cv2.flip(in_image, 0)
    horizontal_flip              = cv2.flip(in_image, 1)
    vertical_and_horizontal_flip = cv2.flip(in_image, 2)

    images.append(vertical_flip)
    images.append(horizontal_flip)
    images.append(vertical_and_horizontal_flip)


    return images

def detect(data, templates, w, h):
    img_filenames = os.listdir(data)

    root = os.path.abspath(".")
    output_dir = os.path.join(root, "outputs")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for filename in img_filenames:
        img_rgb = cv2.imread(os.path.join(data, filename))

        print(f"Processing file {filename}.")

        img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        #
        # metody dopasowania: TM_SQDIFF_NORMED, TM_CCOEFF_NORMED, TM_CCORR_NORMED (nieunormowane są zbyt czułe) 
        # TM_SQDIFF_NORMED - najlepsze wyniki dla progu 0.8 - 0.85
        # TM_CCOEFF_NORMED - najlepsze wyniki dla 0.4
        # TM_CCORR_NORMED - najlepsze wyniki dla 0.835
        #

        locations = []

        for t in templates:
            res_ccoef  = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            res_sqdiff = cv2.matchTemplate(img, t, cv2.TM_SQDIFF_NORMED)
            res_ccorr  = cv2.matchTemplate(img, t, cv2.TM_CCORR_NORMED)

            threshold_ccoeff = 0.4
            threshold_sqdiff = 0.8
            threshold_ccorr  = 0.835

            loc_ccoeff = np.where(res_ccoef >= threshold_ccoeff)
            loc_sqdiff = np.where(res_sqdiff >= threshold_sqdiff)
            loc_ccorr  = np.where(res_ccorr >= threshold_ccorr)

            locations.append(loc_ccoeff)
            locations.append(loc_sqdiff)
            locations.append(loc_ccorr)

        #
        # usuwanie nadmiarowych boxów
        #

        boundings = []

        for loc in locations:
            for pt in zip(*loc[::-1]):
                if len(boundings) == 0:
                    boundings.append(pt)
                else:
                    collision = False
                    for b in boundings:
                        if abs(pt[0] - b[0]) < 0.9*w and abs(pt[1] - b[1]) < 0.9*h:
                            collision = True
                            break
                    if not collision:
                        boundings.append(pt)
    
        print(f"Detected {len(boundings)} objects.")
        for b in boundings:
            cv2.rectangle(img_rgb, b, (b[0] + w, b[1] + h), (0,0,255), 3)
        
        cv2.imwrite(os.path.join(output_dir, "output_" + filename), img_rgb)
        
        

root_dir = os.path.abspath(".")
data_dir = os.path.join(root_dir, "data")

template = cv2.imread('car.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]

templates = performAugmentation(template)

detect(data_dir, templates, w, h)
