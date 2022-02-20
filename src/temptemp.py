from timeit import default_timer as timer
import cv2 as cv
from copy import deepcopy as cp

img = cv.imread('/Users/david/workspace/palm-beesly/test_img/sample2.png')

start = timer()
copied = img.copy()
end = timer()
print(end - start)

start = timer()
copied = cp(img)
end = timer()
print(end - start)
