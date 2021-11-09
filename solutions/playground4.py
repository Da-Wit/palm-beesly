import cv2
import numpy as np
import utils

#
# The goal of code: implementing high pass filter
#
# [DONE] TODO understand and document what high pass filter is.
# [DONE] TODO implement high pass filter
# TODO (optional) implement adjustable high pass filter


# All below codes are implementing of sobel filter V1

import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.2.png"
img = cv2.imread(image_path)
cv2.imshow("img", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

cv2.imshow('Sobel Image', grad)

cv2.waitKey(0)
cv2.destroyAllWindows()
