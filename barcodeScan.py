import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

sobelX = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
sobelY = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

gradient = cv2.subtract(sobelX, sobelY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("Gradient", gradient)
cv2.waitKey(0)

(T, thresh) = cv2.threshold(gradient, 175, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholding", thresh)
cv2.waitKey(0)

# Rectangular kernel has longer length than height to join barcode's black bars
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
# MORPH_CLOSE is dilation followed by erosion
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey(0)

# Erosion removes white spots
closed = cv2.erode(closed, None)
closed = cv2.dilate(closed, None)
cv2.imshow("Erode and Dilate", closed)
cv2.waitKey(0)


# Find contours and sort them from greatest to smallest
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# Largest contour
c = sorted(cnts, key=cv2.contourArea)[-1]

# Minimum bounding box for contour
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)

cv2.drawContours(image, [box], -1, (255, 255, 51), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)