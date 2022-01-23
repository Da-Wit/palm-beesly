import cv2 as cv
import numpy as np

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


# Example of function zoom
# img = cv.imread('image_path')
# cv.imshow('original.png', img)
# zoomed = zoom(img, 0.5)
# cv.imwrite('zoomed.png', zoomed)

# Example of function threshold
# img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#
# if img is None:
#     print("Image is empty!!")
#     exit(1)
#
# cv.imshow("threshold", threshold(img))