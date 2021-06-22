import cv2
import utils

image = cv2.imread("C:/Users/USER/workspace/palm/images/sample3.png")

image = utils.ResizeWithAspectRatio(image,height=678)
image = utils.remove_bground(image)
canny = utils.canny(image)
cv2.imshow("image",image)
cv2.imshow("canny",canny)
cv2.waitKey(0)
