import cv2
import numpy as np

def nothing(x):
    pass

img = cv2.imread("5_histogram.jpg", cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow('Binarization', cv2.WINDOW_NORMAL)

cv2.createTrackbar('H-Low', 'Binarization', 0, 255, nothing)
cv2.createTrackbar('S-Low', 'Binarization', 0, 255, nothing)
cv2.createTrackbar('V-Low', 'Binarization', 0, 255, nothing)

cv2.createTrackbar('H-Upp', 'Binarization', 255, 255, nothing)
cv2.createTrackbar('S-Upp', 'Binarization', 255, 255, nothing)
cv2.createTrackbar('V-Upp', 'Binarization', 255, 255, nothing)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    HL = cv2.getTrackbarPos('H-Low', 'Binarization')
    SL = cv2.getTrackbarPos('S-Low', 'Binarization')
    VL = cv2.getTrackbarPos('V-Low', 'Binarization')
    lower = (HL,SL,VL)

    HU = cv2.getTrackbarPos('H-Upp', 'Binarization')
    SU = cv2.getTrackbarPos('S-Upp', 'Binarization')
    VU = cv2.getTrackbarPos('V-Upp', 'Binarization')
    upper = (HU,SU,VU)

    bin_img = cv2.inRange(hsv, lower, upper)
    cv2.imshow('Binarization', bin_img)

cv2.destroyAllWindows()
