import cv2
import numpy as np
import utils

# Load image
image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.png"

original = cv2.imread(image_path)
img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

img = cv2.equalizeHist(img)

img = cv2.GaussianBlur(img, (9, 9), 0)

img = cv2.Canny(img, 40, 80)
cv2.imshow('Canny', img)

lined = np.copy(original) * 0
lines = cv2.HoughLinesP(img, 1, np.pi / 180, 15, np.array([]), 50, 20)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(lined, (x1, y1), (x2, y2), (0, 0, 255))

output = cv2.addWeighted(original, 0.8, lined, 1, 0)
cv2.imshow('original', original)
cv2.imshow('lined', lined)
cv2.waitKey(0)
cv2.destroyAllWindows()
