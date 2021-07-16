import cv2
import something
import utils
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
image_path = "C:/Users/USER/workspace/palm/images/sample4.png"
image = cv2.imread(image_path)
image = utils.resize(image, height=600)
image = utils.remove_bground(image)


# cv2.imshow("original", image)

mp_palm = utils.get_palm_original(image, mp_hands, mp_drawing)
cv2.imshow("palm_ori", mp_palm)

weird_palm = something.get_palm(image)
cv2.imshow("weird_palm", weird_palm)



# hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# img2 = utils.remove_bground(image)
# adaptive_threshold = utils.adaptive_threshold(img2)
# canny = utils.canny(image)

# cv2.imshow("adaptive_threshold", adaptive_threshold)


cv2.waitKey(0)
