# Import image
import cv2
import numpy as np
import utils


# Load image
image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.2.png"
img = cv2.imread(image_path)
cv2.imshow("img", img)

img = cv2.medianBlur(img, 3)  # Add median filter to image
img = utils.remove_bground(img)

# img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

height = img.shape[0]
width = img.shape[1]

blank_image = np.zeros((height, width, 3), np.uint8)

blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2YCR_CB)
imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)


# print(img_roi)
for i in range(height):
    for j in range(width):
        Y, Cr, Cb = imgYCC[i][j]
        if 64 < Y < 256 and 129 < Cr < 256 and 24 < Cb < 256:
            blank_image[i][j] = imgYCC[i][j]

img_bgr = cv2.cvtColor(blank_image, cv2.COLOR_YCrCb2BGR)
cv2.imshow("img_bgr", img_bgr)


gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
sobel = utils.sobel(equalized)

cv2.imshow("sobel", sobel)


cv2.waitKey(0)
cv2.destroyAllWindows()
