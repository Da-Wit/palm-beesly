import cv2
import numpy as np
import calculator


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
    return cv2.Canny(blur, 10, 120, apertureSize=3)


def get_distance(coord1, coord2):
    return ((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)**(1/2)


def get_degree(coord1, coord2):
    return (coord1[1] - coord2[1])/((coord1[0] - coord2[0])+10**(-9))


def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    return cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)


def threshold(image):
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    return ret, thresh

# coord1 is closer to cnt than coord2


def aws(image, cnt, coord1, coord2):
    degree = get_degree(coord1, coord2)
    # luxk is a coordinate on a linear equation
    # that includes both of coord1 and coord2.
    # Distance of luxk-coord1 and
    # distance of coord1-coord2 are same.
    # Also, luxk is closer to coord1 than coord2.
    luxk = np.array([2*coord1[0]-coord2[0], 2*coord1[1]-coord2[1]])

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


def aws_new(image, cnt, coord1, coord2):
    degree = get_degree(coord1, coord2)
    # luxk is a coordinate on a straight line
    # that includes both of coord1 and coord2.
    # Distance of luxk-coord1 and
    # distance of coord1-coord2 are same.
    # Also, luxk is closer to coord1 than coord2.
    luxk = np.array([2*coord1[0]-coord2[0], 2*coord1[1]-coord2[1]])

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


# mediapipe 라이브러리를 써서 손에 있는 특정 부위들의
# 좌표값을 구해주는 함수
# 그 좌표값들을 반환함
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


# mediapipe 라이브버리 예제(https://google.github.io/mediapipe/solutions/hands.html)
# 에서 사용된 기본 손 감지 코드
def get_palm_original(image, mp_hands, mp_drawing):
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
        img = cv2.flip(img, 1)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return img


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

    part_of_contour = contour[smaller_index:bigger_index+1]

    return part_of_contour, isCoord1IndexBiggerThanCoord2Index
