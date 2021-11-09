import cv2
import numpy as np
import math

# 이미지의 비율을 유지하며 높이, 너비 중 하나를 인수로 받아,
# 적절하게 이미지 크기를 조절하는 함수입니다.


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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


def canny(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    equalized = cv2.equalizeHist(denoised)
    blur = cv2.GaussianBlur(equalized, (9, 9), 0)
    min_thresholding = 100
    max_thresholding = 200
    return cv2.Canny(blur, threshold1=min_thresholding, threshold2=max_thresholding)


def get_distance(coord1, coord2):
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**(1 / 2)


# 두 좌표를 인수로 받아서 그 두 좌표를 지나는 선의 기울기를 구하는 함수
# 기울기를 구하는  분모가 0이 될 수 있어서, 분모에 1e-9(적당히 작은 값)를 더했다.


def get_degree(coord1, coord2):
    return (coord1[1] - coord2[1]) / ((coord1[0] - coord2[0]) + 10**(-9))


def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    return cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)


def threshold(image):
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower, upper 내부의 값들은 순서대로
    # Hue(색상), Saturation(채도), Value(명도)를 의미하며,
    # 각각의 값의 범위는 0-180, 0-255, 0-255입니다.
    # 손바닥 인식을 위해선 이 값들을 조절하면 됩니다.
    lower = np.array([1, 30, 120], dtype="uint8")
    upper = np.array([20, 200, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    return ret, thresh

# coord1(좌표1)과 coord2(좌표2)를 인수로 받아서
# 연장선(좌표1과 좌표2 사이의 거리만큼 좌표1 쪽으로
# 연장한 선), 즉, x가 연장선 끝쪽의 점이라고 하면
# x와 좌표1 사이의 거리와 좌표1과 좌표2 사이의 거리가
# 같고 그 둘의 기울기또한 같다.
# 그래서 x와 좌표1을 서로 이웃하지 않는 직사각형의
# 점으로 놓을 때, 그 직사각형 안의 cnt 좌표들 중,
# 가장 기울기가 좌표1, 좌표2 사이의 기울기와 가까운
# 좌표를 반환한다.
# coord1 is closer to cnt than coord2


def aws(image, cnt, coord1, coord2):
    degree = get_degree(coord1, coord2)
    # luxk is a coordinate on a linear equation
    # that includes both of coord1 and coord2.
    # Distance of luxk-coord1 and
    # distance of coord1-coord2 are same.
    # Also, luxk is closer to coord1 than coord2.
    luxk = np.array([2 * coord1[0] - coord2[0], 2 * coord1[1] - coord2[1]])

    if coord1[0] > luxk[0]:
        bigger_x = coord1[0]
        smaller_x = luxk[0]
    else:
        bigger_x = luxk[0]
        smaller_x = coord1[0]

    if coord1[1] > luxk[1]:
        bigger_y = coord1[1]
        smaller_y = luxk[1]
    else:
        bigger_y = luxk[1]
        smaller_y = coord1[1]

    min_degree_gap = 999999999999
    coord_of_min_degree_gap = np.array([0, 0])
    for [[cnt_x, cnt_y]] in cnt:
        if cnt_x <= bigger_x and cnt_x >= smaller_x and cnt_y <= bigger_y and cnt_y >= smaller_y:
            degree_gap = abs(get_degree(
                np.array([cnt_x, cnt_y]), coord1) - degree)
            if degree_gap < min_degree_gap:
                min_degree_gap = degree_gap
                coord_of_min_degree_gap = np.array([cnt_x, cnt_y])
    return coord_of_min_degree_gap

# coord1 is closer to cnt than coord2

# aws와 거의 유사하지만 aws는 가장 기울기 차가
# 작은 cnt 좌표를 반환하지만 이 함수는 일정 수준
# 보다 더 기울기 차가 작을 경우 해당하는 좌표들을
# 모두 한 배열에 넣은 뒤, 그 배열 중 좌표1과의
# 거리가 가장 가까운 좌표를 반환한다.
# 하지만 aws_new는 현재 사용되지는 않는다.


def aws_new(image, cnt, coord1, coord2):
    degree = get_degree(coord1, coord2)
    # luxk is a coordinate on a straight line
    # that includes both of coord1 and coord2.
    # Distance of luxk-coord1 and
    # distance of coord1-coord2 are same.
    # Also, luxk is closer to coord1 than coord2.
    luxk = np.array([2 * coord1[0] - coord2[0], 2 * coord1[1] - coord2[1]])

    if coord1[0] > luxk[0]:
        bigger_x = coord1[0]
        smaller_x = luxk[0]
    else:
        bigger_x = luxk[0]
        smaller_x = coord1[0]

    if coord1[1] > luxk[1]:
        bigger_y = coord1[1]
        smaller_y = luxk[1]
    else:
        bigger_y = luxk[1]
        smaller_y = coord1[1]

    degree_gap_criteria = 0.08
    coords_on_line = np.zeros((0, 2), dtype=np.int32)
    count = 0

    for [[cnt_x, cnt_y]] in cnt:
        if cnt_x <= bigger_x and cnt_x >= smaller_x and cnt_y <= bigger_y and cnt_y >= smaller_y:
            count += 1
            degree_gap = abs(get_degree(
                np.array([cnt_x, cnt_y]), coord1) - degree)
            if degree_gap < degree_gap_criteria:
                coords_on_line = np.append(
                    coords_on_line, [[cnt_x, cnt_y]], axis=0)
                # coord_of_min_degree_gap = np.array([cnt_x, cnt_y])

    # if len(coords_on_line) == 0:
    #     raise Exception('Length of coords_on_line is 0. Fix this function.')
    #     return -1
    min_distance = 999
    result = np.zeros((2), dtype=np.int32)
    for coord in coords_on_line:
        distance = get_distance(coord, coord1)
        if distance < min_distance:
            min_distance = distance
            result = coord

    return result


# 제일 큰 contours를 구해주는 함수
# max contour를 구하는 이유는
# background에 손 말고 이상한 것들이 남아있는 경우가 많음
# 그것들을 지우고 손만 남기기 위함
def get_max_contour(contours):
    max = 0
    maxcnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(max < area):
            max = area
            maxcnt = cnt
    return maxcnt


# 배경을 지우고 손만 남겨주는 함수
# 배경을 검정색으로 해줌
# 손 주위 완벽히 지워지지 않고 뭔가 남을 수 있음
def remove_bground(image):
    ret, thresh = threshold(image)

    # 경계선 찾음
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 영역 찾기
    maxcnt = get_max_contour(contours)
    mask = np.zeros(thresh.shape).astype(thresh.dtype)

    # 경계선 내부 255로
    cv2.fillPoly(mask, [maxcnt], [255, 255, 255])
    return cv2.bitwise_and(image, image, mask=mask)


# mediapipe 라이브러리를 써서 손에 있는
# 특정 부위들의 좌표값을 반환하는 함수
# mediapipe 라이브버리 예제는 다음 링크 참고!
# https://google.github.io/mediapipe/solutions/hands.html
def get_hand_form(image, mp_hands):
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
        # Flip image around y-axis for correct handedness output
        img = cv2.flip(image, 1)

        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return None
        return results.multi_hand_landmarks[0].landmark


def get_contour(image):
    ret, thresh = threshold(image)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return get_max_contour(contours)


def get_part_of_contour(image, contour, coord1, coord2):
    # make iterable coords tuple to execute for loop
    coords = (coord1, coord2)
    indices_of_coords = np.zeros((2, 3), dtype=np.int32)

    for index in range(len(coords)):
        coord = coords[index]

        # indices having same value with coord
        ihsvwc = np.asarray(np.where(contour == coord)).T
        for i in ihsvwc:
            if coord[0] == contour[i[0]][i[1]][0] and coord[1] == contour[i[0]][i[1]][1]:
                indices_of_coords[index] = i

    if indices_of_coords[0][0] < indices_of_coords[1][0]:
        smaller_index = indices_of_coords[0][0]
        bigger_index = indices_of_coords[1][0]
        isCoord1IndexBiggerThanCoord2Index = False
    else:
        smaller_index = indices_of_coords[1][0]
        bigger_index = indices_of_coords[0][0]
        isCoord1IndexBiggerThanCoord2Index = True

    part_of_contour = contour[smaller_index:bigger_index + 1]

    if isCoord1IndexBiggerThanCoord2Index != True:
        part_of_contour = np.flipud(part_of_contour)

    return part_of_contour


def center(points):
    M = cv2.moments(points)
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
    (oldY, oldX) = img.shape
    # rotate about center of image.
    M = cv2.getRotationMatrix2D(
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

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
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
    return (x2, y2)


def sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

    img_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

    # If you decrease the value of weight like 0.5, sub-lines would
    # be removed. If decrease, it increase and hard to notice main lines.
    default_weight = 1
    img_sobel = cv2.addWeighted(
        img_sobel_x, default_weight, img_sobel_y, default_weight, 0)
    return img_sobel


def custom_sobel(im, xKernel, yKernel):
    # This function is incomplete. Do not use it.

    # # // Call using built-in Sobel

    # out1 = cv2.convertScaleAbs(out1.copy())

    # // Create custom kernel
    # xVals = np.array([0.125, 0, -0.125, 0.25, 0, -0.25,
    #                  0.125, 0, -0.125]).reshape(3, 3)

    xFiltered = cv2.filter2D(im, -1, xKernel, None,
                             (-1, -1), 0, cv2.BORDER_DEFAULT)
    xFiltered = cv2.convertScaleAbs(xFiltered.copy())

    yFiltered = cv2.filter2D(im, -1, yKernel, None,
                             (-1, -1), 0, cv2.BORDER_DEFAULT)
    yFiltered = cv2.convertScaleAbs(yFiltered.copy())

    img_sobel = cv2.addWeighted(xFiltered, 1, yFiltered, 1, 0)
    return img_sobel
