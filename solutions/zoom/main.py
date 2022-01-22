import cv2 as cv

def zoom(img, zoom_factor=1.5):
    return cv.resize(img, None, fx=zoom_factor, fy=zoom_factor)

# Example
# img = cv.imread('image_path')
# cv.imshow('original.png', img)
# zoomed = zoom(img, 0.5)
# cv.imwrite('zoomed.png', zoomed)
