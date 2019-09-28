from __future__ import division

import cv2
import numpy as np
import math
import time
from matplotlib import patches as mpatches
from skimage import segmentation,measure



# color
color_R = (0, 0, 255)

# color tH
lower_Red1 = np.array([0,100,100])
upper_Red1 = np.array([10,255,255])
lower_Red2 = np.array([160,100,100])
upper_Red2 = np.array([180,255,255])

# size tH
area_min = 30
area_max = 2000


# percent tH
pL = 0.25 
pR = 0.75
pD = 0.5
#pU = 0


# cir tH
cir_1 = 0.06
cir_2 = 0.6

class LaneDetector:
    def __init__(self, road_horizon, road_bottom, FBDL, FBDR, prob_hough=True):
        self.prob_hough = prob_hough
        self.vote = 50
        self.roi_theta = 0.3
        self.road_horizon = road_horizon
        self.road_bottom = road_bottom
        self.FBDL = FBDL
        self.FBDR = FBDR

    def _standard_hough(self, img, init_vote):
        # Hough transform wrapper to return a list of points like PHough does
        lines = cv2.HoughLines(img, 1, np.pi/180, init_vote)
        points = [[]]
        for l in lines:
            for rho, theta in l:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*a)
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*a)
                points[0].append((x1, y1, x2, y2))
        return points

    def _base_distance(self, x1, y1, x2, y2, width):
        # compute the point where the give line crosses the base of the frame
        # return distance of that point from center of the frame
        if x2 == x1:
            return (width*0.5) - x1
        m = (y2-y1)/(x2-x1)
        c = y1 - m*x1
        base_cross = -c/m
        return (width*0.5) - base_cross

    def _scale_line(self, x1, y1, x2, y2, frame_height):
        # scale the farthest point of the segment to be on the drawing horizon
        if x1 == x2:
            if y1 < y2:
                y1 = self.road_horizon
                y2 = self.road_bottom
                return x1, y1, x2, y2
            else:
                y2 = self.road_horizon
                y1 = self.road_bottom
                return x1, y1, x2, y2
        if y1 < y2:
            m = (y1-y2)/(x1-x2)
            x1 = ((self.road_horizon-y1)/m) + x1
            y1 = self.road_horizon
            x2 = ((self.road_bottom-y2)/m) + x2
            y2 = self.road_bottom
            m = (y2-y1)/(x2-x1)
            x2 = ((self.road_horizon-y2)/m) + x2
            y2 = self.road_horizon
            x1 = ((self.road_bottom-y1)/m) + x1
            y1 = self.road_bottom
        return x1, y1, x2, y2


    def detectLight(self, frame):
        img = frame
        size = img.shape
        cimg = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, lower_Red1, upper_Red1)
        mask2 = cv2.inRange(hsv, lower_Red2, upper_Red2)
        mask_R = cv2.add(mask1, mask2)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))

        opened_R  = cv2.morphologyEx(mask_R, cv2.MORPH_OPEN, element)

        segmentation.clear_border(opened_R)  
    
    
        label_image_R = measure.label(opened_R)  
        borders_R = np.logical_xor(mask_R, opened_R)
        label_image_R[borders_R] = -1
        
        for region_R in measure.regionprops(label_image_R):
            if region_R.convex_area < area_min or region_R.area > area_max:# or region_R.area > 2000:
                continue
     
            minr, minc, maxr, maxc = region_R.bbox

            area = region_R.area                  
            convex_area  = region_R.convex_area   
            perimeter = region_R.perimeter    
            diameter = np.maximum(maxr-minr,maxc-minc)

            # flit     
            if perimeter == 0:
                circularity = 1
            else:
                circum_circularity = convex_area / (diameter * diameter)
                circularity = area / (perimeter * perimeter) 

            #if inRange(minc, minr, maxc, maxr, size[1], size[0]) and (circularity >= cir_1 and circum_circularity >= cir_2):
            if (circularity >= cir_1 and circum_circularity >= cir_2):
                return 1   

        # is red, cvout-img
        return 0
        

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roiy_end = frame.shape[0]
        roix_end = frame.shape[1]
        roi = img[self.road_horizon:self.road_bottom, 0:roix_end]
        blur = cv2.medianBlur(roi, 5)
        contours = cv2.Canny(blur, 60, 120)

        if self.prob_hough:
            lines = cv2.HoughLinesP(contours, 1, np.pi/180, self.vote, minLineLength=30, maxLineGap=100)
        else:
            lines = self.standard_hough(contours, self.vote)

        if lines is not None:
            # find nearest lines to center
            lines = lines+np.array([0, self.road_horizon, 0, self.road_horizon]).reshape((1, 1, 4))  # scale points from ROI coordinates to full frame coordinates
            left_bound = None
            right_bound = None
            for l in lines:
                # find the rightmost line of the left half of the frame and the leftmost line of the right half
                for x1, y1, x2, y2 in l:
                    theta = np.abs(np.arctan2((y2-y1), (x2-x1)))  # line angle WRT horizon
                    if theta > self.roi_theta:  # ignore lines with a small angle WRT horizon
                        dist = self._base_distance(x1, y1, x2, y2, frame.shape[1])
                        if left_bound is None and dist < 0:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is None and dist > 0:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
                        elif left_bound is not None and 0 > dist > left_dist:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is not None and 0 < dist < right_dist:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
            if left_bound is not None:
                left_bound = self._scale_line(left_bound[0], left_bound[1], left_bound[2], left_bound[3], frame.shape[0])
            if right_bound is not None:
                right_bound = self._scale_line(right_bound[0], right_bound[1], right_bound[2], right_bound[3], frame.shape[0])

            return [left_bound, right_bound]

