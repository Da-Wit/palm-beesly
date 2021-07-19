import cv2
import something
import utils
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
image_path = "C:/Users/USER/workspace/palm/images/sample2.png"
image = cv2.imread(image_path)
image = utils.resize(image, height=600)
# TODO make it work whatever height is
# Now, this module only work well when
# the height of image is 600.

# TODO make remove_all_except_hand function,
# Before removing background,
# remove everything except hand areas.
# If you don't, invoking remove_bground function
# may remove hand.
image = utils.remove_bground(image)
cv2.imshow("original", image)

mp_palm = utils.get_palm_original(image, mp_hands, mp_drawing)
cv2.imshow("palm_ori", mp_palm)

palm = something.get_palm(image)
cv2.imshow("palm", palm)


# hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# img2 = utils.remove_bground(image)
# adaptive_threshold = utils.adaptive_threshold(img2)
# canny = utils.canny(image)

# cv2.imshow("adaptive_threshold", adaptive_threshold)


cv2.waitKey(0)
