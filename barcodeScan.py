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

(T, thresh) = cv2.threshold(gradient, 175, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholding", thresh)


