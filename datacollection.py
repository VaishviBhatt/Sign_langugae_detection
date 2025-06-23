import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0) #using laptop camera if using web camera put 0
detector = HandDetector(maxHands=1) #hand detection only for 1 hand
offset = 20
image_size = 300
counter = 0

folder = "/Users/vaishvibhatt/Desktop/projects/sign_language_detection/data/Hello"

while True:
    success , img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imageWhite = np.ones((image_size,image_size,3),np.unit8)*255

        imageCrop = img[y-offset : y + h + offset, x - offset: x + w + offset]

        imageCropSHape = imageCrop.shape

        aspectratio = h/w

        if aspectratio > 1:
            k = image_size / h
            weightCal = math.ceil(k*w)
            imageResize = cv2.resize(imageCrop, weightCal, image_size)
            imageResizeShape = imageResize.shape
            




