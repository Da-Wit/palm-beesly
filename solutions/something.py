import cv2
import mediapipe as mp
import numpy as np
import utils
import calculator

# 숫자가 크면 클수록 pip와 mcp 사이의 간격이 커집니다.
# 소수도 가능하고, 에러가 나면 이 수를 키우거나 줄이면 됩니다.
# DFER: Default Finger Expansion Ratio
DFER = 6

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


def get_coord(x, y, width, height):
    return np.array([int(x * width), int(y * height)])


def get_finger_coord(target, neighbor, width, height, ratio=DFER):
    return get_coord(
        target.x + ((neighbor.x - target.x) / ratio),
        target.y + ((neighbor.y - target.y) / ratio),
        width,
        height)


def get_intersection_coord_between_finger_and_palm(pip_mp, mcp_mp, img, pip_ratio=DFER, mcp_ratio=DFER):
    height, width, _ = img.shape
    pip = get_finger_coord(pip_mp, mcp_mp, width, height, pip_ratio)
    mcp = get_finger_coord(mcp_mp, pip_mp, width, height, mcp_ratio)
    X, Y = 0, 1
    binary_img = utils.adaptive_threshold(img)
    ret, thresh = utils.threshold(img)

    # y1 must be bigger than y2
    y1, y2, x1, x2 = 0, 0, 0, 0
    if(mcp[Y] > pip[Y]):
        y1, x1 = mcp[Y], mcp[X]
        y2, x2 = pip[Y], pip[X]
    else:
        y1, x1 = pip[Y], pip[X]
        y2, x2 = mcp[Y], mcp[X]
    return calculator.get_intersection(binary_img, x1, y1, x2, y2)


def get_coords(landmark, img):
    height, width, _ = img.shape
    thumb_outside = get_coord(
        landmark[THUMB_MCP].x, landmark[THUMB_MCP].y, width, height)
    thumb_mid = get_finger_coord(
        landmark[THUMB_CMC], landmark[THUMB_MCP], width, height, ratio=2)
    thumb_inside = get_finger_coord(
        landmark[THUMB_CMC], landmark[WRIST], width, height, ratio=200)
    wrist = get_coord(landmark[WRIST].x, landmark[WRIST].y, width, height)
    middle_between_thumb_wrist = get_coord(
        landmark[THUMB_CMC].x, landmark[THUMB_CMC].y, width, height)

    pinky = get_intersection_coord_between_finger_and_palm(landmark[PINKY_PIP],
                                                           landmark[PINKY_MCP],
                                                           img)
    ring = get_intersection_coord_between_finger_and_palm(landmark[RING_FINGER_PIP],
                                                          landmark[RING_FINGER_MCP],
                                                          img)
    middle = get_intersection_coord_between_finger_and_palm(landmark[MIDDLE_FINGER_PIP],
                                                            landmark[MIDDLE_FINGER_MCP],
                                                            img)
    index = get_intersection_coord_between_finger_and_palm(landmark[INDEX_FINGER_PIP],
                                                           landmark[INDEX_FINGER_MCP],
                                                           img)
    index_inside = get_coord(
        landmark[INDEX_FINGER_MCP].x, landmark[INDEX_FINGER_MCP].y, width, height)

    return thumb_outside, thumb_mid, thumb_inside, wrist, middle_between_thumb_wrist, pinky, ring, middle, index, index_inside


def get_palm(image):
    img = image.copy()
    landmark = utils.get_hand_form(img, mp_hands)
    if not landmark:
        return None

    thumb_outside, thumb_mid, thumb_inside, wrist, middle_between_thumb_wrist, pinky, ring, middle, index, index_inside = get_coords(
        landmark, img)

    palm_coords = np.array(
        [thumb_mid, wrist, pinky, ring, middle, index], dtype=np.int32)

    cnt = utils.get_contour(img)

    aws = utils.aws(img, cnt, pinky, ring)
    aws2 = utils.aws(img, cnt, wrist, thumb_mid)

    # aws3 = utils.aws_new(img, cnt, index, middle)

    aws4 = utils.aws(img, cnt, thumb_outside, index_inside)
    aws5 = utils.aws(img, cnt, thumb_mid, index_inside)
    aws6 = utils.aws(img, cnt, thumb_inside, index_inside)

    part_of_contour, isFingerIndexBiggerThanHandBottomIndex = utils.get_part_of_contour(
        img, cnt, aws, aws2)

    coords = np.array([[pinky], [ring], [middle], [
                      index], [aws4], [aws5], [aws6]])

    if isFingerIndexBiggerThanHandBottomIndex:
        part_of_contour = np.append(part_of_contour, coords, axis=0)
    else:
        part_of_contour = np.append(part_of_contour, np.flipud(coords), axis=0)
    cv2.polylines(img, [part_of_contour], True, (255, 255, 0), 2)

    # ret, thresh = utils.threshold(img)
    # mask = np.zeros(thresh.shape).astype(thresh.dtype)
    # cv2.fillPoly(mask, [part_of_contour], [255, 255, 255])
    # img = cv2.bitwise_and(img, img, mask=mask)

    return img
