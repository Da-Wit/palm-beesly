import cv2
import mediapipe as mp
import numpy as np
import utils
import calculator

mp_hands = mp.solutions.hands

# 숫자가 크면 클수록 pip와 mcp 사이의 간격이 커집니다.
# 소수도 가능하고, 에러가 나면 이 수를 키우거나 줄이면 됩니다.
EXTREME_CUTTING_RATIO = 6
EXTRA_LENGTH_EXCEPT_FINGERS = 2

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def init_finger_coords(pip_coords, mcp_coords, width, height):
    copied_pip = pip_coords.copy()
    copied_mcp = mcp_coords.copy()
    for i in range(len(pip_coords)):
        X, Y = 0, 1

        pip_y, pip_x = copied_pip[i][Y], copied_pip[i][X]
        mcp_y, mcp_x = copied_mcp[i][Y], copied_mcp[i][X]

        copied_pip[i] = (
            int((pip_x + ((mcp_x - pip_x) / EXTREME_CUTTING_RATIO)) * width),
            int((pip_y + ((mcp_y - pip_y) / EXTREME_CUTTING_RATIO)) * height),
        )
        copied_mcp[i] = (
            int((mcp_x + ((pip_x - mcp_x) / EXTREME_CUTTING_RATIO)) * width),
            int((mcp_y + ((pip_y - mcp_y) / EXTREME_CUTTING_RATIO)) * height),
        )
    copied_pip = np.array(copied_pip, dtype=np.int32)
    copied_mcp = np.array(copied_mcp, dtype=np.int32)
    return (copied_pip, copied_mcp)

def get_palm(image):
    cnt = utils.get_contour(image)
    landmark = utils.get_hand_form(image, mp_hands)
    if not landmark:
        return None

    img = image.copy()
    image_height, image_width, _ = img.shape

    wrist_coord = (
        int(landmark[mp_hands.HandLandmark.WRIST].x * image_width),
        int(landmark[mp_hands.HandLandmark.WRIST].y * image_height)
    )

    thumb_coord = (
             int((landmark[mp_hands.HandLandmark.THUMB_CMC].x + landmark[mp_hands.HandLandmark.THUMB_MCP].x) / 2 * image_width),
             int((landmark[mp_hands.HandLandmark.THUMB_CMC].y + landmark[mp_hands.HandLandmark.THUMB_MCP].y) / 2 * image_height)
    )


    # fingers are ordered by pinky, ring, middle, index
    # In finger, pip is the fourth joint and mcp is the third one.
    pip_coords = np.array(
        [
            (
                landmark[mp_hands.HandLandmark.PINKY_PIP].x,
                landmark[mp_hands.HandLandmark.PINKY_PIP].y,
            ),
            (
                landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            ),
            (
                landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            ),
            (
                landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            )
        ], np.float64
    )
    mcp_coords = np.array(
        [
                         (
                             landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                             landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                         ),
            (
                             landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                             landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                         ),
            (
                             landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                             landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                         ),
            (
                             landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                             landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                         )
        ], np.float64
    )


    # The reason why I plus 2(EXTRA_LENGTH_EXCEPT_FINGERS) is because palm_coords
    # Includes the coordinates of thumb and wrist
    # Not just fingers' coordinates
    length_of_palm_coods = len(pip_coords) + EXTRA_LENGTH_EXCEPT_FINGERS
    palm_coords = np.empty((length_of_palm_coods, 2), dtype=np.int32)
    palm_coords[0] =  thumb_coord
    palm_coords[1] = wrist_coord

    pip_coords, mcp_coords = init_finger_coords(
        pip_coords, mcp_coords, image_width, image_height)

    for i in range(len(pip_coords)):
        palm_coords[i +
                    EXTRA_LENGTH_EXCEPT_FINGERS] = utils.get_finger_intersection(img, pip_coords[i], mcp_coords[i])

    cnt = utils.get_contour(image)
    # img = cv2.polylines(img,[cnt],True,(255,0,255),2)

    pinky = palm_coords[EXTRA_LENGTH_EXCEPT_FINGERS]
    ring = palm_coords[EXTRA_LENGTH_EXCEPT_FINGERS+1]
    middle = palm_coords[EXTRA_LENGTH_EXCEPT_FINGERS+2]
    index = palm_coords[EXTRA_LENGTH_EXCEPT_FINGERS+3]


    aws = utils.aws(image,cnt,pinky,ring)
    aws2 = utils.aws(image,cnt,wrist_coord,thumb_coord)

    # img = cv2.circle(img,pinky,10,(0,255,0),3)
    # img = cv2.circle(img,ring,10,(0,255,0),3)
    # img = cv2.circle(img,aws,10,(0,255,0),3)
    # img = cv2.circle(img,aws2,10,(0,255,0),3)

    part_of_contour, isFingerIndexBiggerThanHandBottomIndex = utils.get_part_of_contour(image,cnt,aws,aws2)
    # img = cv2.circle(img,aws,3,(255,0,0),5)

    img = cv2.polylines(img,[part_of_contour],False,(255,255,0),1)
    nparray_thumb = np.array([thumb_coord])
    nparray_wrist = np.array([wrist_coord])


    if isFingerIndexBiggerThanHandBottomIndex:
        part_of_contour = np.insert(part_of_contour,0,nparray_thumb, axis=0)
        part_of_contour = np.insert(part_of_contour,1,nparray_wrist, axis=0)
    else:
        part_of_contour = np.append(part_of_contour, np.array([nparray_wrist,nparray_thumb]), axis=0)
        img = cv2.polylines(img,[part_of_contour],False,(255,255,0),1)
    img = cv2.polylines(img,[part_of_contour],False,(255,255,0),1)




    # print("part_of_contour: ",part_of_contour)
    for palm_coord in palm_coords:
        img = cv2.circle(img,palm_coord,3,(255,0,0),1)

    # img = cv2.bitwise_and(img, img, mask=mask)


    return img
