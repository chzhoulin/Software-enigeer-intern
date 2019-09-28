# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:46:02 2018

@author: Administrator
"""

import cv2
import numpy as np
from skimage import segmentation, measure

area_min = 10
area_max = 2000

def detect(frame):
    #gamma_frame = np.power(frame, 0.8)
    YUV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y = cv2.split(YUV_frame)[0]
    y[y > 100] =255
    y[y < 100] = 0
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))

    opened_y  = cv2.morphologyEx(y, cv2.MORPH_OPEN, element)
    segmentation.clear_border(opened_y)  
    
    label_image_y = measure.label(opened_y)  
    borders_y = np.logical_xor(y, opened_y)
    label_image_y[borders_y] = -1

    region = []

    for region_y in measure.regionprops(label_image_y):
        if len(measure.regionprops(label_image_y)) == 0:
            return 0
            
        if region_y.convex_area < area_min or region_y.area > area_max:# or region_R.area > 2000:
            continue
        minr, minc, maxr, maxc = region_y.bbox
        area = region_y.area
        #perimeter = region_y.perimeter
        #diameter = max(maxr - minr,maxc - minc)
        if (maxc-minc)/(maxr - minr)<=0.8 and (maxc-minc)/(maxr-minr)>=0.2:
            if region_y.convex_area/area > 0.8 and region_y.convex_area/area < 1:
                region.append([minr, minc, maxr, maxc])
    
    r = cv2.split(frame)[2]
    g = cv2.split(frame)[1]
    r_g = r-g
    #g_r = g-r
    y_rg = np.round(1/(0.008+0.01*np.power(np.e, (-0.1*(r_g-70)))))
    #y_gr = np.round(1/(0.008+0.01*np.power(np.e, (-0.1*(g_r-70)))))

    opened_rg  = cv2.morphologyEx(y_rg, cv2.MORPH_OPEN, element)
    segmentation.clear_border(opened_rg)  
    
    label_image_rg = measure.label(opened_rg)
    borders_rg = np.logical_xor(y_rg, opened_rg)
    label_image_rg[borders_rg] = -1
    
    for region_rg in measure.regionprops(label_image_rg):
        mean = np.mean(frame[minr:maxr,minc:maxc])
        print(mean)
        if mean > 100: #thershold should be changed
            return 1
        else:
            return 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        red_flag = detect(frame)
        cv2.imshow("demo", frame)
        print(red_flag)
        #cv2.waitKey(30)

        if cv2.waitKey(30) & 0xFF == ord('q'):   # q: quit
            break  

cap.release()  
cv2.destroyAllWindows()
        