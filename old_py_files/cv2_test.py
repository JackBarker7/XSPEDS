import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = np.load("data/images/image8.npy")
img_gray = img / img.max() * 255
img_gray = img_gray.astype(np.uint8)

img_gray = cv.medianBlur(img_gray, 5)

circles = cv.HoughCircles(
    img_gray,
    cv.HOUGH_GRADIENT,
    1,
    20,
    param1=50,
    param2=30,
    minRadius=5000,
    maxRadius=10000,
)

print(circles)

cv.imshow("img_gray", img_gray)
cv.waitKey(0)
cv.destroyAllWindows()
