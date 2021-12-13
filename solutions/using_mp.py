import cv2
import mediapipe as mp
import numpy as np

import calculator
import utils

# 숫자가 크면 클수록 pip와 mcp 사이의 간격이 커집니다.
# 소수도 가능하고, 에러가 나면 이 수를 키우거나 줄이면 됩니다.
# 이 값을 조절하면 엄지를 제외한
# 나머지 손가락의 좌표를 조절하게 됩니다.
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

# 아래 get_palm에서 리턴하는 coords함수의 인덱스들
PINKY = 0
RING = 1
MIDDLE = 2
INDEX = 3
THUMB_TOP = 4
THUMB_MID = 5
THUMB_BOTTOM = 6


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

    # y1 must be bigger than y2
    y1, y2, x1, x2 = 0, 0, 0, 0
    if (mcp[Y] > pip[Y]):
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
    thumb_inside = get_coord(
        landmark[THUMB_CMC].x, landmark[THUMB_CMC].y, width, height)
    # get_finger_coord(
    #     landmark[THUMB_CMC], landmark[WRIST], width, height, ratio=200)
    # thumb_inside = get_finger_coord(
    #     landmark[THUMB_CMC], landmark[WRIST], width, height, ratio=200)
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


def refine_coords(coords):
    refined_coords = np.zeros((0, 1, 2), dtype=np.int32)
    for coord in coords:
        if (coord[0][0] != 0 and coord[0][1] != 0):
            refined_coords = np.append(refined_coords, [coord], axis=0)
    return refined_coords


def get_palm(image):
    img = image.copy()
    landmark = utils.get_hand_form(img, mp_hands)
    if not landmark:
        return None
    img = utils.remove_bground(img)
    thumb_outside, thumb_mid, thumb_inside, wrist, middle_between_thumb_wrist, pinky, ring, middle, index, index_inside = get_coords(
        landmark, img)

    palm_coords = np.array(
        [thumb_mid, wrist, pinky, ring, middle, index], dtype=np.int32)

    cnt = utils.get_contour(img)
    img2 = image.copy()
    img2 = cv2.polylines(img2, [cnt], True, (255, 0, 0), 2)

    sonnal_top = utils.aws(img, cnt, pinky, ring)
    sonnal_bottom = utils.aws(img, cnt, wrist, thumb_mid)

    # aws3 = utils.aws_new(img, cnt, index, middle)

    thumb_top = utils.aws(img, cnt, thumb_outside, index_inside)
    thumb_mid = utils.aws(img, cnt, thumb_mid, index_inside)
    thumb_bottom = utils.aws(img, cnt, thumb_inside, index_inside)

    # sonnal[0]이 손날 위쪽(새끼손까락쪽),
    # sonnal[len(sonnal)-1]이 손날 아래쪽(손목쪽)
    sonnal = utils.get_part_of_contour(img, cnt, sonnal_top, sonnal_bottom)

    coords = np.array([[pinky], [ring], [middle], [index], [thumb_top], [thumb_mid], [thumb_bottom]])

    refined_coords = refine_coords(coords)

    # sonnal = np.append(sonnal, refined_coords, axis=0)

    return sonnal, coords


def get_landmark_coord(landmark, width, height):
    return get_coord(landmark.x, landmark.y, width, height)


def get_hand_landmark(image):
    landmark = utils.get_hand_form(image, mp_hands)
    if not landmark:
        return None
    hand_landmarks = [
        landmark[WRIST],
        landmark[THUMB_CMC],
        landmark[THUMB_MCP],
        landmark[THUMB_IP],
        landmark[THUMB_TIP],
        landmark[INDEX_FINGER_MCP],
        landmark[INDEX_FINGER_PIP],
        landmark[INDEX_FINGER_DIP],
        landmark[INDEX_FINGER_TIP],
        landmark[MIDDLE_FINGER_MCP],
        landmark[MIDDLE_FINGER_PIP],
        landmark[MIDDLE_FINGER_DIP],
        landmark[MIDDLE_FINGER_TIP],
        landmark[RING_FINGER_MCP],
        landmark[RING_FINGER_PIP],
        landmark[RING_FINGER_DIP],
        landmark[RING_FINGER_TIP],
        landmark[PINKY_MCP],
        landmark[PINKY_PIP],
        landmark[PINKY_DIP],
        landmark[PINKY_TIP],
    ]
    height, width = image.shape[:2]

    for i in range(len(hand_landmarks)):
        hand_landmarks[i] = get_landmark_coord(hand_landmarks[i], width, height)

    return hand_landmarks
