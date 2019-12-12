import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./input/haarcascades/haarcascade_frontalface_default.xml')
# face detection e khode open-cv
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(type(faces))
    if type(faces) is tuple:
        return None
    # for (x, y, w, h) in faces:
    #     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img
