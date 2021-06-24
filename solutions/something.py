import cv2
import mediapipe as mp
import numpy as np
import utils
import calculator
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 숫자가 크면 클수록 pip와 mcp 사이의 간격이 커집니다.
# 소수도 가능하고, 에러가 나면 이 수를 키우거나 줄이면 됩니다.
EXTREME_CUTTING_RATIO = 6

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


def init_finger_coords(pip_coords,mcp_coords,width,height):
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
    copied_pip = np.array(copied_pip,dtype=np.int32)
    copied_mcp = np.array(copied_mcp,dtype=np.int32)
    return (copied_pip,copied_mcp)


def get_palm(image):
    landmark = utils.get_hand_form(image)
    if not landmark:
      return None

    img = image.copy()
    image_height, image_width, _ = img.shape

    wrist_coord = (
              int(landmark[mp_hands.HandLandmark.WRIST].x * image_width),
              int(landmark[mp_hands.HandLandmark.WRIST].y * image_height)
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
                          ]
                          ,np.float64
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
                        ]
                        ,np.float64
                        )

    length_of_palm_coods = len(pip_coords)+1
    palm_coords = np.empty((length_of_palm_coods,2),dtype=np.int32)
    palm_coords[0] = wrist_coord

    pip_coords, mcp_coords = init_finger_coords(pip_coords,mcp_coords,image_width,image_height)


    for i in range(len(pip_coords)):
        palm_coords[i+1] = utils.get_intersection(img,pip_coords[i],mcp_coords[i])


    ret,thresh = utils.threshold(img)
    mask = np.zeros(thresh.shape).astype(thresh.dtype)
    cv2.fillPoly(mask, [palm_coords], [255, 255, 255])
    img = cv2.bitwise_and(img,img,mask = mask)

    return img













#######################    legacy codes below    #######################














# import cv2
# import mediapipe as mp
# import utils
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
#
#
#
#
# def makeCircle(img,y,x,radius):
#     circle_coordinates = (x,y)
#     color = (255, 0, 0)
#     thickness = 2
#     return cv2.circle(img,circle_coordinates,radius,color,thickness)
#
#
#
# # For static images:
# IMAGE_FILES = [
#     "C:/Users/USER/workspace/palm/images/sample1.png",
#     "C:/Users/USER/workspace/palm/images/sample2.png",
#     "C:/Users/USER/workspace/palm/images/sample3.png",
#     "C:/Users/USER/workspace/palm/images/sample4.png"
# ]
#
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     image = utils.remove_bground(image)
#
#
#     # resize image's height 600 fixing the ratio
#     image = utils.resize(image,height=600)
#
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     print("results:",results)
#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       circle_coordinates = [
#         (
#             (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y + hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)/2,
#             (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x + hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x)/2
#         ),
#         (
#             (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y + hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)/2,
#             (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x + hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x)/2
#         ),
#         (
#             (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y + hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y)/2,
#             (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x + hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x)/2
#         ),
#         (
#         (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y + hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y)/2,
#         (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x + hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x)/2
#         ),
#         (
#             hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
#             hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
#         ),
#         (
#             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
#             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x
#         ),
#         (
#             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
#             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
#         ),
#       ]
#     ret,thresh = utils.threshold(annotated_image)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     c = max(contours, key=cv2.contourArea)
#
#     cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
#
#
#
#     cv2.imshow("img" + str(idx),cv2.flip(image,1))
#     cv2.waitKey(0)
