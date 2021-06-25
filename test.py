from matte import bayesian_matte

import cv2

img1 = cv2.imread("blonde.png", cv2.IMREAD_COLOR)

trimap = cv2.imread("trimap.jpg", cv2.IMREAD_GRAYSCALE)

matte = bayesian_matte(img1, trimap, 25, 8, 10)
print(matte)