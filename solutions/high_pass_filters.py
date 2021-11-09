import cv2
import numpy as np
import utils


#
# The goal of code: implementing high pass filter
#
# [DONE] TODO understand and document about high pass filter
# [DONE] TODO implement sobel filter
# [DONE] TODO (optional) implement adjustable sobel filter
# TODO understand and document about sobel filter
# TODO implement Canny
# TODO understand and document about Canny
# TODO (optional) implement adjustable Canny


# sobel filter
def using_sobel():
    image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.png"
    img = cv2.imread(image_path)

    sobel = utils.sobel(img)
    cv2.imshow("utils sobel", sobel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
