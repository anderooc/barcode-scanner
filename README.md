# OpenCV Project - Barcode Scanner 
(Finished) This program takes an image that contains a barcode and draws a box around the barcode

Chapters Used
1. Chapter 3 - Loading and display of images
2. Chapter 4 - The minimum bounding box has color (51, 255, 255) 
3. Chapter 6 - Image subtraction (cv2.subtract) between SobelX and SobelY to represent the gradient. Also grayscales the image to more esaily split into black and white
4. Chapter 8 - Gaussian Blur (cv2.GaussianBlur) of image to smooth out the gradient representation by removing noise
5. Chapter 9 - Use simple thresholding to split all pixels up into either having a value of 0 or 255.
6. Chapter 10 - Use Sobel method for gradient representation. Instead of using a bitwise function though, cv2.subtract combines the X and Y representations.
7. Chapter 11 - Use contours to find the largest outline (which is likely the barcode)
