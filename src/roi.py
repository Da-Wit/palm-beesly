import cv2 as cv
import mediapipe as mp
import numpy as np
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


# 좌표를 저장할 때 사용할 클래스
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


def get_outta_sonnal_top(ring, pinky_mcp, pinky_pip):
    x = pinky_mcp.x + (pinky_mcp.x - ring.x) * 0.9
    y = pinky_mcp.y + (pinky_mcp.y - pinky_pip.y) * 0.2

    return Point(int(x), int(y))


def get_outta_sonnal_bottom(ring, pinky_mcp, pinky_pip):
    x = pinky_mcp.x + (pinky_mcp.x - ring.x) * 0.9
    y = pinky_mcp.y + (pinky_mcp.y - pinky_pip.y) * 0.8

    return Point(int(x), int(y))


def get_index_left(index_mcp, middle_mcp, index_mid):
    x = index_mcp.x + (index_mcp.x - middle_mcp.x) * 0.8
    y = (index_mcp.y * 0.8) + (index_mid.y * 0.2)

    return Point(int(x), int(y))


def crop(img_param, pts_li, black_background):
    pts = np.array(pts_li, np.int32)

    rect = cv.boundingRect(pts)
    x, y, w, h = rect
    croped = img_param[y:y + h, x:x + w].copy()

    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)

    dst = cv.bitwise_and(croped, croped, mask=mask)
    if black_background:
        return dst, rect
    bg = np.ones_like(croped, np.uint8) * 255
    cv.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst
    return dst2, rect


def get_palm_roi(img_param):
    img = img_param.copy()
    height, width = img.shape[:2]
    landmarks = get_landmarks(img)

    if landmarks is None:
        print(ERROR_MESSAGE["MEDIAPIPE_CANNOT_DETECT_LANDMARK_OF_HAND"])
        exit(1)

    landmark_points = calculate_landmark_points(landmarks, height, width)

    index_mid = middle_finger(landmark_points, INDEX_FINGER_MCP)
    middle_mid = middle_finger(landmark_points, MIDDLE_FINGER_MCP)
    ring_mid = middle_finger(landmark_points, RING_FINGER_MCP)

    ring_mcp = landmark_points[RING_FINGER_MCP]
    pinky_mcp = landmark_points[PINKY_MCP]
    pinky_pip = landmark_points[PINKY_PIP]

    index_mcp = landmark_points[INDEX_FINGER_MCP]
    middle_mcp = landmark_points[MIDDLE_FINGER_MCP]

    thumb_mcp = landmark_points[THUMB_MCP]
    wrist = landmark_points[WRIST]

    outta_sonnal_top = get_outta_sonnal_top(ring_mcp, pinky_mcp, pinky_pip)
    outta_sonnal_bottom = get_outta_sonnal_bottom(ring_mcp, pinky_mcp, pinky_pip)

    index_left = get_index_left(index_mcp, middle_mcp, index_mid)

    roi = [index_mid,
           middle_mid,
           ring_mid,
           outta_sonnal_top,
           outta_sonnal_bottom,
           wrist,
           thumb_mcp,
           index_left]
    roi = [p.tuple() for p in roi]
    cropped, rect = crop(img, roi, black_background=True)

    return cropped, rect


if __name__ == "__main__":
    for i in range(30):
        print(f"i : {i}")
        image = init_img(i)
        roi = get_palm_roi(image)
        cv.imshow('roi', roi)
        cv.waitKey(0)
