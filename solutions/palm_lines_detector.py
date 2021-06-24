import cv2
import something
import utils
import numpy as np

image_path = "C:/Users/USER/workspace/palm/images/sample3.png"

image = cv2.imread(image_path)
image = utils.resize(image, height=600)

cv2.imshow("originalss.png", image)
image = something.get_palm(image)
cv2.imshow("palm_only1.png", image)

cv2.waitKey(0)
