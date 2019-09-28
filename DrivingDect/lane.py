'''
# lane lines
@ E.C.Ares 2017.11.02
! MIT licence
~ cv + threshold
'''

from __future__ import division

import sys
import cv2

import track
import detect
# from drstrdrv import *

import socket
import struct
import time
import random
import math
from datetime import datetime
import darknet as dn
import numpy as np


# lane_soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# server_addr = ("127.0.0.1", 9005)

slopeMax = 999
slopeL = 0.3
slopeR = 0.3
# upline = 300    # change1
upline = 380
# 和视频高度一致
lolane = 480  # don't move
Xl = 200
Xr = 440

leftXR = 380
leftXL = 200
rightXL = 260  # rightXL <= leftXR
rightXR = 440

''' no tH
leftXR = 640
leftXL = 0
rightXL = 0    # rightXL <= leftXR
rightXR = 640
'''


def main():
    cap = cv2.VideoCapture(0)  # 0
    cv2.namedWindow("demo")  # a new window
    ticks = 0
    s = 1
    t = 0
    lt = track.LaneTracker(2, 0.1, 200)
    ld = detect.LaneDetector(upline, lolane, Xl, Xr)
    ip_port = ('192.168.1.176',50002)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0',50000))#UDP服务器端口和IP绑定
    #buf, addr = sock.recvfrom(40960)

    sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock1.bind(('0.0.0.0',50001))#UDP服务器端口和IP绑定
    #buf1, addr1 = sock1.recvfrom(40960)

    sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock2.bind(('0.0.0.0',50002))#UDP服务器端口和IP绑定
    #buf2, addr2 = sock2.recvfrom(40960)
    
    dn.set_gpu(1)
    net = dn.load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = dn.load_meta(b"cfg/coco.data")

    datad = 0.01
    pres=0
    ns=0
    speed=0
    ndis=0

    while (1):
        precTick = ticks
        ticks = cv2.getTickCount()
        dt = (ticks - precTick) / cv2.getTickFrequency()
        # print(dt)
        ret, frame = cap.read()
	
        # object detection using yolo

        res = dn.detect_numpy(net, meta, image=frame)
        ndis=0
        for item in res:
            pt1 = (int(item[2][0]-item[2][2] * 0.5), int(item[2][1]-item[2][3]*0.5))
            pt2 = (int(item[2][0]+item[2][2]*0.5), int(item[2][1]+item[2][3]*0.5))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 4)
            cv2.putText(frame, str(item[0], encoding="utf8"), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 4)            
            ns = item[2][3] 
            ndis = max(pt1[1],pt2[1])

        speed = ns-pres
        pres = ns
        
        print('speed:',speed)
        print('ndis:',ndis)
        
        # cv2.rectangle(frame, (0,0), (599, 402), (0, 255, 0), 4)
        # frame2 = frame[upline:640, 0:480]

        if (ret):
            lanes = ld.detect(frame)
            # print(lanes)
            # if lanes:
            predicted = lt.predict(dt)

            if lanes and (t > 1):  # predict succeed
                sys.exit()
            # if s:
            #  print(ld.detectLight(frame))

            t = t + 1

            if predicted is not None and (predicted[0][1][0] == predicted[0][1][0]):
                if (t > 0):
                    t = t - 1

                xa, ya, xb, yb = predicted[0]
                x1, y1, x2, y2 = predicted[1]

                print('1: ',xa,' ',ya)
                print('2: ',xb,' ',yb)
                print('3: ',x1,' ',y1)
                print('4: ',x2,' ',y2)

                pt1=[138,480]
                pt2=[265,380]
                pt3=[550,480]
                pt4=[370,380]

                ptc=np.float32([pt1,pt2,pt3,pt4])

                ptt1=[200,450]
                ptt2=[200,300]
                ptt3=[500,450]
                ptt4=[500,300]

                pttc=np.float32([ptt1,ptt2,ptt3,ptt4])

                matrix = cv2.getPerspectiveTransform(ptc,pttc)

                print('matrix:',matrix)

                nimg = cv2.warpPerspective(frame,matrix,(650,500))
                if float(xb - xa) == 0:
                    kl = -slopeMax
                else:
                    kl = float(yb - ya) / float(xb - xa)
                if float(x2 - x1) == 0:
                    k1 = slopeMax
                else:
                    k1 = float(y2 - y1) / float(x2 - x1)
                X = (xb * x2 - x1 * xa) / (xb + x2 - x1 - xa)
                # print("x")
                # print(xb, X, x1)    #left m right   ->   X

                # data = X + 320
                data = (xb + x2) / 2
                # data = drstrdrv(data,)
                fYaw = 300-5 - data[0]
                if (abs(fYaw) < 30):
                    fYaw = fYaw * 0.2
                elif (abs(fYaw) < 60):
                    fYaw = fYaw * 0.8
                else:
                    fYaw = fYaw * 0.9
                datac = float(math.atan(fYaw / 100)) / 2

                if (datad >= 0.25):
                    datad = 0.08

                elif (abs(fYaw) > 23):

                    datad = 0.25 + (abs(fYaw) / 120.0)
                    # data = drstrdrv(data,speed)
                # data = random.random()
                if ndis>380 and ndis< 420:
                    datad += 0.35
                print('datad::', datad)
                print(fYaw)
                # data = 0.5
                data2 = X

                sock2.sendto(struct.pack('<dd', datac,datad),  ip_port)
                #sock2.sendto(struct.pack('<d', datad),  addr2)

                #time.sleep(0.01)
                # sock.sendto(data,("192.168.1.107", 9005))
                cv2.imshow('tp',nimg)
                '''
                print 'L'
                print ((xa[0], ya[0]), (xb[0], yb[0]),kl)
                print 'R'
                print ((x1[0], y1[0]), (x2[0], y2[0]),k1)
                '''
                if (kl < -slopeL):
                    # if float(xb) < leftXR and float(xb) > leftXL:
                    cv2.line(frame, (predicted[0][0], predicted[0][1]), (predicted[0][2], predicted[0][3]), (0, 0, 255),
                             5)
                if (k1 > slopeR):
                    # if float(x1) < leftXR and float(x1) > leftXL:
                    cv2.line(frame, (predicted[1][0], predicted[1][1]), (predicted[1][2], predicted[1][3]), (0, 0, 255),
                             5)
            lt.update(lanes)

            # print(frame.shape)
            cv2.imshow('demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            s = 1 - s
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 1
            break


if __name__ == '__main__':
    main()
