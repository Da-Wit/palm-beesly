import cv2
import something
import utils

image_path = "C:/Users/USER/workspace/palm/images/sample2.png"

image = cv2.imread(image_path)
image = utils.resize(image, height=600)

cv2.imshow("original", image)
image = something.get_palm(image)
cv2.imshow("palm", image)
image = utils.canny(image)

cv2.imshow("canny", image)

cv2.waitKey(0)
