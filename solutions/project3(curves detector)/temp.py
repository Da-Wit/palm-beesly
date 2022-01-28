import cv2

path = "C:/Users/think/Desktop/temp/vert_thresh.png"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("ori", img)
rst = cv2.ximgproc.thinning(img)
cv2.imshow("after", rst)
cv2.waitKey(0)
