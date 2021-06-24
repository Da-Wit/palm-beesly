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
    return cv2.Canny(blur, 10, 150, apertureSize=3)


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
        # print("results:",results.multi_handedness)

        if not results.multi_hand_landmarks:
            return None
        return results.multi_hand_landmarks[0].landmark

# 손가락 3번째 마디 부분의 좌표값들을 지닌 배열(coord2)과
# 3번째 마디 약간 아래의 손바닥 부분의 좌표값들을 지닌
# 배열(coord1)을 이용해서 coord1과 coord2의 각각의
# 좌표들로 직선을 만들어서 그 직선 안에서
# 손가락 3번째 마디가 끝나는, 3번째 마디와
# 손바닥 부분의 경계의 좌표를 구해서
# 반환하는 함수


def get_intersection(image, coord1, coord2):
    X, Y = 0, 1
    binary_image = adaptive_threshold(image)

    # y1 is bigger than y2
    y1, y2, x1, x2 = 0, 0, 0, 0
    if(coord2[Y] > coord1[Y]):
        y1, x1 = coord2[Y], coord2[X]
        y2, x2 = coord1[Y], coord1[X]
    else:
        y1, x1 = coord1[Y], coord1[X]
        y2, x2 = coord2[Y], coord2[X]
    return calculator.get_intersection(binary_image, x1, y1, x2, y2)
