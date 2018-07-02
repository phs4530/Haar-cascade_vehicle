# -*- coding: utf-8 -*-
import numpy as np
import cv2

print(cv2.__version__)

cascade_src = 'cars.xml'
video_src = 'C:\image\sample_video6.mp4'
#video_src = 'dataset/video2.avi'

def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

ret, image = cap.read()
height, width = image.shape[:2] # 이미지 높이, 너비
vertices = np.array([[(0,height*0.8),(width*0.2, height*0.3), (width*0.8, height*0.3), (width,height*0.8)]], dtype=np.int32)


while True:
    ret, image = cap.read()
    if (type(image) == type(None)):
        break

    roi_img = region_of_interest(image, vertices)  # vertices에 정한 점들 기준으로 ROI 이미지 생성
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('video', image)
    cv2.imshow('gray', gray)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
