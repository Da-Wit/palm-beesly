import cv2 as cv
import numpy as np
import copy

def get_roi(img, min_grayscale=0):
    copied = copy.deepcopy(img)
    height, width = copied.shape[:2]

    topmost = 0
    downmost = height - 1
    leftmost = 0
    rightmost = width - 1

    done = False

    for x in range(width):
        if done is True:
            break
        for y in range(height):
            if copied[y][x] > min_grayscale:
                leftmost = x
                done = True

    done = False

    for y in range(height):
        if done is True:
            break
        for x in range(width):
            if copied[y][x] > min_grayscale:
                topmost = y
                done = True

    done = False

    for y in range(height - 1, -1, -1):
        if done is True:
            break
        for x in range(width):
            if copied[y][x] > min_grayscale:
                downmost = y
                done = True

    done = False

    for x in range(width - 1, -1, -1):
        if done is True:
            break
        for y in range(height):
            if copied[y][x] > min_grayscale:
                rightmost = x
                done = True

    return copied[topmost:downmost, leftmost:rightmost]

def zoom(img, zoom_factor=1.5):
    return cv.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def threshold(img_param):
    height, width = img_param.shape[:2]

    result = np.zeros((height, width, 1), dtype=np.uint8)

    kernel = np.ones((3, 3)) / (3 ** 2)
    blured = cv.filter2D(img_param, -1, kernel)

    for x in range(width):
        for y in range(height):
            if img_param[y][x] > blured[y][x]:
                result[y][x] = 255
            else:
                result[y][x] = 0

    return result


image_path = "C:/Users/think/workspace/palm-beesly/test_img/sample5.4.png"
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None:
    print("Image is empty!!")
    exit(1)

img = get_roi(img)


cv.imshow("origi", img)
cv.imshow("threshold", threshold(img))
cv.waitKey(0)
# Example
# img = cv.imread('image_path')
# cv.imshow('original.png', img)
# zoomed = zoom(img, 0.5)
# cv.imwrite('zoomed.png', zoomed)
