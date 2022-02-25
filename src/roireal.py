import cv2 as cv
import mediapipe as mp
import numpy as np
from copy import deepcopy as cp
import utils
from constants import ERROR_MESSAGE

mp_hands = mp.solutions.hands
WRIST = mp_hands.HandLandmark.WRIST
THUMB_CMC = mp_hands.HandLandmark.THUMB_CMC
THUMB_MCP = mp_hands.HandLandmark.THUMB_MCP
THUMB_IP = mp_hands.HandLandmark.THUMB_IP
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_MCP = mp_hands.HandLandmark.INDEX_FINGER_MCP
INDEX_FINGER_PIP = mp_hands.HandLandmark.INDEX_FINGER_PIP
INDEX_FINGER_DIP = mp_hands.HandLandmark.INDEX_FINGER_DIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_FINGER_MCP = mp_hands.HandLandmark.MIDDLE_FINGER_MCP
MIDDLE_FINGER_PIP = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
MIDDLE_FINGER_DIP = mp_hands.HandLandmark.MIDDLE_FINGER_DIP
MIDDLE_FINGER_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
RING_FINGER_MCP = mp_hands.HandLandmark.RING_FINGER_MCP
RING_FINGER_PIP = mp_hands.HandLandmark.RING_FINGER_PIP
RING_FINGER_DIP = mp_hands.HandLandmark.RING_FINGER_DIP
RING_FINGER_TIP = mp_hands.HandLandmark.RING_FINGER_TIP
PINKY_MCP = mp_hands.HandLandmark.PINKY_MCP
PINKY_PIP = mp_hands.HandLandmark.PINKY_PIP
PINKY_DIP = mp_hands.HandLandmark.PINKY_DIP
PINKY_TIP = mp_hands.HandLandmark.PINKY_TIP


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def tuple(self, inverse=False):
        return (self.x, self.y) \
            if inverse is False \
            else (self.y, self.x)

    def distance_between(self, another_one):
        return utils.distance_between((self.x, self.y), (another_one.x, another_one.y))

    def slope_between(self, another_one):
        return utils.slope_between((self.x, self.y), (another_one.x, another_one.y))


def init_img(i):
    image_path = f"/Users/david/workspace/palm-beesly/sample_img/sample{i}.png"
    img = cv.imread(image_path)
    # 이미지가 제대로 불러와지지 않으면 에러 출력하고 다음 숫자로 넘어감
    if img is None:
        print(ERROR_MESSAGE["IMG_IS_EMPTY"])
        exit(1)
    return img


def get_landmarks(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None
        return results.multi_hand_landmarks[0].landmark


def get_coord(x, y, width, height):
    return np.array([int(x * width), int(y * height)])


def calculate_landmark_points(landmarks, height, width):
    points = []
    for l in landmarks:
        x = round(l.x * width)
        y = round(l.y * height)

        points.append(Point(x, y))
    return points


def middle_finger(landmark_points_param, first_idx):
    if len(landmark_points_param) < (first_idx + 2):
        print(ERROR_MESSAGE["SHORT_LANDMARK_POINTS_PARAM"])
        exit(1)
    first_point = landmark_points_param[first_idx]
    second_point = landmark_points_param[first_idx + 1]

    x = ((first_point.x * 0.6) + (second_point.x * 0.4))
    y = ((first_point.y * 0.6) + (second_point.y * 0.4))

    result_point = int(x), int(y)
    return Point(*result_point)


def blur_equalizehist_adap_thresh(bgr_img):
    gray = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
    adaptive_threshold = utils.adaptive_threshold(blurred, 11, -7)
    return adaptive_threshold


def get_point_of_outta_sonnal1(ring, pinky_mcp, pinky_pip):
    x = pinky_mcp.x + (pinky_mcp.x - ring.x) * 0.9
    y = pinky_mcp.y + (pinky_mcp.y - pinky_pip.y) * 0.2

    return Point(int(x), int(y))


def get_point_of_outta_sonnal2(ring, pinky_mcp, pinky_pip):
    x = pinky_mcp.x + (pinky_mcp.x - ring.x) * 0.9
    y = pinky_mcp.y + (pinky_mcp.y - pinky_pip.y) * 0.8

    return Point(int(x), int(y))


if __name__ == "__main__":
    for i in range(33):
        print(f"i : {i}")
        original = init_img(i)
        # cv.imshow('original', original)

        copied = original.copy()
        height, width = copied.shape[:2]
        landmarks = get_landmarks(copied)

        if landmarks is None:
            print(ERROR_MESSAGE["MEDIAPIPE_CANNOT_DETECT_LANDMARK_OF_HAND"])
            exit(1)

        landmark_points = calculate_landmark_points(landmarks, height, width)

        index_mid = middle_finger(landmark_points, INDEX_FINGER_MCP)
        middle_mid = middle_finger(landmark_points, MIDDLE_FINGER_MCP)
        ring_mid = middle_finger(landmark_points, RING_FINGER_MCP)
        pinky_mid = middle_finger(landmark_points, PINKY_MCP)

        ring_mcp = landmark_points[RING_FINGER_MCP]
        pinky_mcp = landmark_points[PINKY_MCP]
        pinky_pip = landmark_points[PINKY_PIP]

        outta_sonnal1 = get_point_of_outta_sonnal1(ring_mcp, pinky_mcp, pinky_pip)
        outta_sonnal2 = get_point_of_outta_sonnal2(ring_mcp, pinky_mcp, pinky_pip)

        for p in (index_mid, middle_mid, ring_mid, outta_sonnal1, outta_sonnal2):
            cv.circle(copied, p.tuple(), 1, (255, 255, 255), 4)
            cv.circle(copied, p.tuple(), 4, (0, 0, 0), 2)

        for p in landmark_points:
            cv.circle(copied, p.tuple(), 1, (255, 255, 255), 4)
            cv.circle(copied, p.tuple(), 4, (0, 0, 0), 2)

        cv.imshow('copied', copied)
        k = cv.waitKey(0)
        # for문 도중 Esc를 누르면 프로그램이 종료되게 함
        if k in (ord('q'), ord('Q'), 66, 27):  # 66 indicates ㅂ, 27 does esc
            exit(0)
