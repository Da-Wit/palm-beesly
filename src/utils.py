import cv2 as cv
import math
import numpy as np


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


def distance_between(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** (1 / 2)


# 두 좌표를 인수로 받아서 그 두 좌표를 지나는 선의 기울기를 구하는 함수
# 기울기를 구하는  분모가 0이 될 수 있어서, 분모에 1e-9(적당히 작은 값)를 더했다.
def slope_between(coord1, coord2):
    return (coord1[1] - coord2[1]) / ((coord1[0] - coord2[0]) + 10 ** (-9))


def adaptive_threshold(img_param, box_size, constant):
    result = cv.adaptiveThreshold(img_param, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, box_size, constant)
    return result


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


# 배경을 지우고 손만 남겨주는 함수
# 배경을 검정색으로 해줌
# 손 주위 완벽히 지워지지 않고 뭔가 남을 수 있음
def remove_bground(image):
    ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 경계선 찾음
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # 가장 큰 영역 찾기
    maxcnt = get_max_contour(contours)
    mask = np.zeros(thresh.shape).astype(thresh.dtype)

    # 경계선 내부 255로
    cv.fillPoly(mask, [maxcnt], [255, 255, 255])
    return cv.bitwise_and(image, image, mask=mask)


def get_contour(image):
    ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return get_max_contour(contours)


def get_center_of_mass(points):
    M = cv.moments(points)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    return [cX, cY]


def getAngle(start, end):
    start_x, start_y = start
    end_x, end_y = end

    d_y = end_y - start_y
    d_x = end_x - start_x
    angle = math.atan(d_y / d_x) * (180.0 / math.pi)

    if d_x < 0.0:
        angle += 180.0
    elif d_y < 0.0:
        angle += 360.0
    return angle


def rotateAndScale(img, scaleFactor=0.5, degreesCCW=30):
    # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    (oldY, oldX) = img.shape[:2]
    # rotate by center of image.
    M = cv.getRotationMatrix2D(
        center=(oldX / 2, oldY / 2), angle=degreesCCW, scale=scaleFactor)

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX),
                  abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)

    # third column of matrix holds translation, which takes effect after rotation.
    M[0, 2] += tx
    M[1, 2] += ty

    print("M", M)

    rotatedImg = cv.warpAffine(img, M, dsize=(int(newX), int(newY)))
    return rotatedImg


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def rotate_point(point, pivot, degree):
    deg = (math.pi / 180) * degree
    x1, y1 = point
    x0, y0 = pivot
    x2 = round(((x1 - x0) * math.cos(deg)) - ((y1 - y0) * math.sin(deg)) + x0)
    y2 = round(((x1 - x0) * math.sin(deg)) + ((y1 - y0) * math.cos(deg)) + y0)
    return x2, y2


def find_index(list_param, condition_funct, multiple=False):
    result = []
    for index in range(len(list_param)):
        if condition_funct(index, list_param[index]) is True:
            if multiple is True:
                result.append(index)
            else:
                return index
    if multiple is True:
        return result
    else:
        return None


def find_indices(list_param, key):
    result = []
    for index in range(len(list_param)):
        if key(index, list_param[index]) is True:
            result.append(index)
    return result
