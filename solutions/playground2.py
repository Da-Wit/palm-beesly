# Import image
import cv2
import numpy as np


# Load image
image_path = f"C:/Users/USER/workspace/palm/images/sample{43}.png"
img = cv2.imread(image_path)
imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
cv2.imshow("ycc", imgYCC)

height = img.shape[0]
width = img.shape[1]
img_roi = np.zeros([height, width, 3])

# print(img_roi)
for i in range(height):
    for j in range(width):
        Y, Cr, Cb = imgYCC[i][j]
        if 64 < Y < 256 and 129 < Cr < 256 and 24 < Cb < 256:
            img_roi[i][j] = imgYCC[i][j]


cv2.imshow("img_roi", img_roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
