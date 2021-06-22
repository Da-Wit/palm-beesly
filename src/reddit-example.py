import cv2
import numpy as np

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


file = "C:/Users/USER/workspace/palm/images/sample1.png"
imageHeight = 600

original = cv2.imread(file)

img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

resize = ResizeWithAspectRatio(img,height=imageHeight)
cv2.imshow("gray",resize)
cv2.waitKey(0)


img = cv2.equalizeHist(img)
resize = ResizeWithAspectRatio(img,height=imageHeight)
cv2.imshow("equalize",resize)
cv2.waitKey(0)

img = cv2.GaussianBlur(img, (9, 9), 0)
resize = ResizeWithAspectRatio(img,height=imageHeight)
cv2.imshow("blur",resize)
cv2.waitKey(0)

img = cv2.Canny(img, 40, 80)
resize = ResizeWithAspectRatio(img,height=imageHeight)
cv2.imshow("canny",resize)
cv2.waitKey(0)

lined = np.copy(original) * 0
lines = cv2.HoughLinesP(img, 1, np.pi / 180, 15, np.array([]), 50, 20)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(lined, (x1, y1), (x2, y2), (0, 0, 255))
resize = ResizeWithAspectRatio(lined,height=imageHeight)
cv2.imshow("lined",resize)
cv2.waitKey(0)

output = cv2.addWeighted(original, 0.8, lined, 1, 0)
resize = ResizeWithAspectRatio(output,height=imageHeight)
cv2.imshow("output",resize)
cv2.waitKey(0)
