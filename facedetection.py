import cv2
import numpy as np
import time

faceDector  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

videoCam = cv2.VideoCapture(0)

if not videoCam.isOpened():
    print("No camera detected")
    exit()

StopPressed = False
while (StopPressed == False):
    ret, vid_skeleton = videoCam.read()

    if ret == True:
        vidGrey = cv2.cvtColor(vid_skeleton, cv2.COLOR_BGR2GRAY)
        faces = faceDector.detectMultiScale(vidGrey, scaleFactor = 1.3, minNeighbors = 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(vid_skeleton, (x, y), (x + w, y + h), (0, 255, 0), 2)

        teks = "Number of face detected = " + str(len(faces))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vid_skeleton, teks, (0, 30), font, 1, (255, 0, 0), 1)

        cv2.imshow("Output", vid_skeleton)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            StopPressed = True
            break


videoCam.release()
cv2.destroyAllWindows()