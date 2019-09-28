import numpy as np
import math  
import cv2  
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
area_min = 10
area_max = 2000


# percent tH
pL = 0.25 
pR = 0.75
pD = 0.5
#pU = 0


# cir tH
cir_1 = 0.06
cir_2 = 0.6



def detect(frame):

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
    diameter = max(maxr - minr,maxc - minc)

    # flit     
    if perimeter == 0:
      circularity = 1
    else:
      circum_circularity = convex_area / (diameter * diameter)
      circularity = area / (perimeter * perimeter) 

    #if inRange(minc, minr, maxc, maxr, size[1], size[0]) and (circularity >= cir_1 and circum_circularity >= cir_2):
    if (circularity >= cir_1 and circum_circularity >= cir_2):
      return 1, 0, 0#, mask_R      

  # is red, cvout-img
  return 0, 0, 0#, mask_R



def inRange(minc, minr, maxc, maxr, x, y):
  if (minc >= x * pL and minc <= x * pR and minr >= y * pD) or (maxc >= x * pL and maxc <= x * pR and maxr >= y * pD):
    return False
  return True



def max(a, b):
  if a>b:
    return a
  else:
    return b



def main():

  cap = cv2.VideoCapture(1)   #1
  #cv2.namedWindow("demo")  # a new window
  
  while(1):  

    ret, frame = cap.read()  # frame in
    if ret:

      red, green, yellow = detect(frame)

      print(red)
      cv2.imshow("demo", frame)
      #cv2.waitKey(30)

      if cv2.waitKey(30) & 0xFF == ord('q'):   # q: quit
      #cv2.imwrite("/home/profiles/result.jpg", frame)  # a pic 
        break  

  cap.release()  
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
