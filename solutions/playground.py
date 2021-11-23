import cv2
import numpy as np
import utils

#
# This code is the code which get a suit rectangle of roi
#


# Load image
image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.2.png"
img = cv2.imread(image_path)

ret, thresh = utils.threshold(img)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

maxcnt = utils.get_max_contour(contours)
mask = np.zeros(thresh.shape).astype(thresh.dtype)
cv2.fillPoly(mask, [maxcnt], [255, 255, 255])

edge = np.zeros(thresh.shape).astype(thresh.dtype)
cv2.drawContours(edge, maxcnt, -1, (255), 1)

filtered_mask = cv2.medianBlur(mask, 17)  # Add median filter to image
x, y, w, h = cv2.boundingRect(maxcnt)

test = np.zeros(thresh.shape).astype(thresh.dtype)
for i in contours:
    cv2.polylines(test, [i], False, (255), 2)
    # cv2.drawContours(test, i, -1, (255), 1)


roi = cv2.bitwise_and(img, img, mask=filtered_mask)
roi = roi[y:y + h, x:x + w]
mask = filtered_mask[y:y + h, x:x + w]
edge = edge[y:y + h, x:x + w]
test = test[y:y + h, x:x + w]


gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

thresh = 127
binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]


cv2.imshow('roi', roi)
cv2.imshow("test", test)
cv2.imshow("mask", mask)
cv2.imshow('gray', gray)
cv2.imshow('binary', binary)
cv2.imshow('edge', edge)


# img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

height = img.shape[0]
width = img.shape[1]

# blank_image = np.zeros((height, width, 3), np.uint8)
#
# blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2YCR_CB)
#
#
# cv2.imshow("img_roi2", blank_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
