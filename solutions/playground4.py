import cv2
import numpy as np
import utils

#
# The goal of code: implementing high pass filter
#
# [DONE] TODO understand and document what high pass filter is.
# TODO implement high pass filter
# TODO (optional) implement adjustable high pass filter


# All below codes are implementing of sobel filter V1

import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.2.png"
img = cv2.imread(image_path)
cv2.imshow("img", img)

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel64f = np.absolute(sobelx64f)

cv2.imshow("Sobel CV_8U", sobelx8u)


cv2.waitKey(0)
cv2.destroyAllWindows()
