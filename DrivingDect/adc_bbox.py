import cv2
import numpy as np
import rospy
import csv
import time
from darknet_ros_msgs.msg import BoundingBoxes
#from darknet_ros_msgs.msg import Image_raw

#orgimg=np.zeros((640, 480, 3))
'''
def image_raw(msg):
    global orgimg
    #chang from this point
    orgimg = msg; 
    #complete

'''

def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - long(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return ct

def coordinate_change(x, y):
    px=x-301
    py=(432-y)
    my=py/24
    mx=px/32
    print("cchange complete!")
    return (mx, my)
    
def bbox_msg(msg):

    """
    Class=[]
    probability=[]
    xmin=[]
    ymin=[]
    xmax=[]
    ymax=[]
    for i in range(len(msg.boundingBoxes)):
      Class.append(msg.boundingBoxes[i].Class)
      probability.append(msg.boundingBoxes[i].probability)
      xmin.append(msg.boundingBoxes[i].xmin)
      ymin.append(msg.boundingBoxes[i].ymin)
      xmax.append(msg.boundingBoxes[i].xmax)
      ymax.append(msg.boundingBoxes[i].ymax)
    print Class
    print probability
    print xmin
    print ymin
    print xmax
    print ymax
    print   
    """
    #get outputfile name
    timesp=get_time_stamp()

    fn1="outputresult"
    filename=fn1+".csv"
    folder="objs"
    filepath=folder+"/"+filename
    out=open(filepath, 'a')
    csv_writer=csv.writer(out)
    
    #show points --YL code
    orgimgwidth = 640.
    orgimglength = 480.
    #global orgimg
    
    imgsize = [480,270]

    p1=[241, 297]
    p2=[590, 329]
    p3=[578, 456]
    p4=[205, 363]
    
    pa1=[round((77.)/orgimgwidth*imgsize[0])+100, round((206.)/orgimglength*imgsize[1])]
    pa2=[round((404.)/orgimgwidth*imgsize[0])+100, round((206)/orgimglength*imgsize[1])]
    pa3=[round((404.)/orgimgwidth*imgsize[0])+100, round((484.)/orgimglength*imgsize[1])]
    pa4=[round((77.)/orgimgwidth*imgsize[0])+100, round((484.)/orgimglength*imgsize[1])]

    #setting cx, cy, which is the zero point of bird eye graph

    pt = np.float32([p1, p2, p3, p4])
    pt2 = np.float32([pa1, pa2, pa3, pa4]) 

    tranMatrix = cv2.getPerspectiveTransform(pt, pt2)
    
    #-------------------------now get a fault matrix to get a perfect blue trangle
    p1f=[256, 290]
    p2f=[384, 290]
    p3f=[404, 304]
    p4f=[236, 304]

    ptf = np.float32([p1f, p2f, p3f, p4f])
    tranMatrix_f = cv2.getPerspectiveTransform(ptf, pt2)

    #------------------------fault matrix got------------------------------


    #preparing blue background for show
    bluecolor = [255, 0, 0] #blue color
    redcolor = [0, 255, 0] #red color
    #backgroundimg = np.zeros((1920, 1080, 3))
    
    #-------------because always something get out, I suspect 640, 480 picture is not really 480, it is narrow
    #backgroundimg = np.zeros((640,480,3))
    backgroundimg = np.zeros((640,620,3))
    #-------------so I make it bigger----------------------------------
    
    backgroundimg[:][:]=bluecolor
    bluebackground = cv2.warpPerspective(backgroundimg, tranMatrix, (imgsize[0]+100, imgsize[1]*2-100), flags=cv2.INTER_LINEAR)
    cv2.imshow('locationimg', bluebackground)
    bluebackground = cv2.imread("background.jpg")
    bluebackground = cv2.resize(bluebackground, (imgsize[0]+100, imgsize[1]*2-100))

    allobjecttext=""
    #get point's location
    for i in range(len(msg.boundingBoxes)):
        pti = [(msg.boundingBoxes[i].xmin + msg.boundingBoxes[i].xmax)/2, msg.boundingBoxes[i].ymax]
        print ("4points:%f, %f, %f, %f"%(msg.boundingBoxes[i].xmin, msg.boundingBoxes[i].ymin, msg.boundingBoxes[i].xmax, msg.boundingBoxes[i].ymax))
        print ("pti is : %f, %f"%(pti[0], pti[1]))
        ptp1 = np.array([[pti]], dtype='float32')
        ptp2 = cv2.perspectiveTransform(ptp1, tranMatrix)

        print("a point location: (%f, %f)\n"%(ptp2[0][0][0], ptp2[0][0][1]))
        (dx, dy) = coordinate_change(ptp2[0][0][0], ptp2[0][0][1])


        centerp = (ptp2[0][0][0], ptp2[0][0][1])
        r = 6
        if msg.boundingBoxes[i].Class=='car':
            showcolor=(0,0,255)
        else:
            showcolor=(0,255,0)
        cv2.circle(bluebackground, centerp, r, showcolor, -1, 3)
        cv2.putText(bluebackground, '{:.3f} {:.3f}'.format(dx, dy), centerp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, showcolor,1)
        allobjecttext=allobjecttext+('%f,%f;'%(dy, dx))

    recordtext="["+allobjecttext+"]"
    list=[timesp, recordtext]
    csv_writer.writerow(list)
            
    bluebackground = cv2.resize(bluebackground, (imgsize[0], imgsize[1]))
    #cv2.rectangle(orgimg, (msg.boundingBoxes[i].xmin, msg.boundingBoxes[i].ymin), (msg.boundingBoxes[i].xmax, msg.boundingBoxes[i].ymax), showcolor, 2)
    #class_name=msg.boundingBoxes[i].Class
    #score=msg.boundingBoxes[i].probability
    #cv2.putText(orgimg, '{:s} {:.3f}'.format(class_name, score), (msg.boundingBoxes[i].xmin, msg.boundingBoxes[i].ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),1)
    #orgimg = cv2.resize(orgimg, (imgsize[0], imgsize[1]))
    #bluebackground = np.column_stack((orgimg, bluebackground))
    cv2.imshow('locationimg', bluebackground)
    cv2.waitKey(2)

def subscribe_bbox():
    rospy.init_node('bbox_sub')
    sub=rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, bbox_msg)
    rospy.spin()
'''
def subscribe_image():
    rospy.init_node('iamge_raw')
    sub=rospy.Subscriber("/darknet_ros/image_raw", BoundingBoxes, bbox_msg)
    rospy.spin()
'''
if __name__=='__main__':
    #subscribe_image()
    subscribe_bbox()
