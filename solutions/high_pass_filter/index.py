import cv2
import numpy as np
import sobel
import scharr

image_path = f"C:/Users/USER/workspace/palm/images/sample{2}.2.png"
img = cv2.imread(image_path)
# cv2.imshow("original", img)

# sobel.using_sobel(img, sobel.SB3X3)
# sobel.using_sobel(img, sobel.SB5X5)
# sobel.using_sobel(img, sobel.SB7X7)
scharr.using_scharr(img, scharr.SC3X3)
scharr.using_scharr(img, scharr.SC5X5)


cv2.waitKey(0)


# 보고를 위한 코드

clicked = False
while not clicked:
    for j in range(1, 11):
        if clicked:
            break
        for i in range(501):
            cv2.setTrackbarPos('scharr', scharr.SC3X3, i)
            cv2.setTrackbarPos('scharr', scharr.SC5X5, i)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                clicked = True
                break
        cv2.setTrackbarPos('gaussian', scharr.SC3X3, j)
        cv2.setTrackbarPos('gaussian', scharr.SC5X5, j)
    clicked = True


cv2.waitKey(0)
cv2.destroyAllWindows()
