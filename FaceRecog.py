import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
faceHaar = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
while True :
    ret, frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces= faceHaar.detectMultiScale(gray,1.2,5,minSize=(20,20))
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        roigray = gray[y:y+h,x:x+w]
        roiimage = frame[y:y+h,x:x+h]
    

    cv.imshow('Color Cam', frame)
    cv.imshow('Gray Cam',gray)

    k = cv.waitKey(30) & 0xff
    if k==27: # Tekan ESC untuk keluar
        break

cap.release
cv.destroyAllWindows()