import cv2 as cv
import math
import numpy as np


# 이미지를 배율로 확대/축소 하는 함수
def zoom(img_param, zoom_factor=None):
    if zoom_factor is None:
        return img_param
    return cv.resize(img_param, None, fx=zoom_factor, fy=zoom_factor)


# 이미지의 비율을 유지하며 높이, 너비 중 하나를 인수로 받아,
# 적절하게 이미지 크기를 조절하는 함수
def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv.resize(image, dim, interpolation=inter)


# 두 좌표 간의 거리를 구함
def distance_between(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** (1 / 2)


# 두 좌표를 인수로 받아서 그 두 좌표를 지나는 선의 기울기를 구하는 함수
# 기울기를 구하는  분모가 0이 될 수 있어서, 분모에 1e-9(적당히 작은 값)를 더했다.
def slope_between(coord1, coord2):
    return (coord1[1] - coord2[1]) / ((coord1[0] - coord2[0]) + 10 ** (-9))


def adaptive_threshold(img_param, box_size, constant):
    result = cv.adaptiveThreshold(img_param, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, box_size, constant)
    return result


def otsu_threshold(img_param):
    ret, otsu = cv.threshold(img_param, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return ret, otsu


# to use this function, you must install opencv-contrib-python
# just paste it: pip install opencv-contrib-python
def thin(gray_img):
    result = cv.ximgproc.thinning(gray_img)
    return result


# 제일 큰 contours를 구해주는 함수
# max contour를 구하는 이유는
# background에 손 말고 이상한 것들이 남아있는 경우가 많음
# 그것들을 지우고 손만 남기기 위함
def get_max_contour(contours):
    max = 0
    maxcnt = None
    for cnt in contours:
        area = cv.contourArea(cnt)
        if max < area:
            max = area
            maxcnt = cnt
    return maxcnt


def get_contour(image):
    ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return get_max_contour(contours)


# 무게중심
def get_center_of_mass(points):
    M = cv.moments(points)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    return [cX, cY]


# 리스트에서 key 콜백함수를 넘겨서 조건을 만족하는
# 값들의 index들을 반환하는 함수
def find_index(list_param, key, multiple=False):
    result = []
    for index in range(len(list_param)):
        if key(index, list_param[index]) is True:
            if multiple is False:
                return index
            result.append(index)

    return result
