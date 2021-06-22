import cv2
import mediapipe as mp
import numpy as np
import utils
import calculator
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
EXTREME_CUTTING_RATIO = 4

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


def get_intersection(image,coord1,coord2):
    X, Y = 0, 1
    binary_image = utils.adaptive_threshold(image)
    print("i caught in a trap")

    # y1 is bigger than y2
    y1, y2, x1, x2 = 0, 0, 0, 0
    if(coord2[Y] > coord1[Y]):
        y1, x1 = coord2[Y], coord2[X]
        y2, x2 = coord1[Y], coord1[X]
    else:
        y1, x1 = coord1[Y], coord1[X]
        y2, x2 = coord2[Y], coord2[X]
    print(calculator.get_intersection)
    return calculator.get_intersection(binary_image, x1, y1, x2, y2)

def init_palm(coords_for_line,width,height):
    for index in range(len(coords_for_line)):
        PIP = 0
        MCP = 1
        X, Y = 0, 1

        pip_y, pip_x = coords_for_line[index][PIP][Y], coords_for_line[index][PIP][X]
        mcp_y, mcp_x = coords_for_line[index][MCP][Y], coords_for_line[index][MCP][X]
        coords_for_line[index] = [
                                  (
                                   int((pip_x + (mcp_x - pip_x) / EXTREME_CUTTING_RATIO) * width),
                                   int((pip_y + (mcp_y - pip_y) / EXTREME_CUTTING_RATIO) * height),
                                  ),
                                  (
                                   int((mcp_x + (pip_x - mcp_x) / EXTREME_CUTTING_RATIO) * width),
                                   int((mcp_y + (pip_y - mcp_y) / EXTREME_CUTTING_RATIO) * height),
                                   )
                                  ]

def calculate_intersection_coordinate(image,intersection_coords,coords_for_line):
    print("in my life")
    result_intersection_coords = intersection_coords.copy()
    for coords in coords_for_line:
        print("i lv more")

        PIP, MCP = 0, 1
        # if(intersection_coord == None):
        # #     # error handling
        intersection_y,intersection_x = get_intersection(image,coords[PIP], coords[MCP])
        print(intersection_y,intersection_x)
        result_intersection_coords = np.concatenate((intersection_coords,np.array([[intersection_x,intersection_y]])))
    print("the lv u make is equal to the love")
    return result_intersection_coords


def get_palm_coordinate(image):
    print("welcome to get_palm_coordinate")
    landmark = utils.get_hand_form(image)
    print("got landmark")

    if not landmark:
      return None
    print("error checekd")

    image_height, image_width, _ = image.shape
    print("got width, height")

    WRIST = [[
              landmark[mp_hands.HandLandmark.WRIST].x * image_width,
              landmark[mp_hands.HandLandmark.WRIST].y * image_height
            ]]
    print("defined WRIST")

    intersection_coords = np.array(WRIST,np.int32)
    print("defined ic")

    coords_for_line = [
                     [
                      (
                       landmark[mp_hands.HandLandmark.PINKY_PIP].x,
                       landmark[mp_hands.HandLandmark.PINKY_PIP].y,
                       ),
                      (
                        landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                        landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                        ),
                      ],
                     [
                      (
                       landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                       landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                       ),
                      (
                        landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                        landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                        ),
                      ],
                     [
                      (
                       landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                       landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                       ),
                      (
                        landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                        landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                        ),
                      ],
                     [
                      (
                       landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                       landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                       ),
                      (
                        landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                        landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                        ),
                      ],
        ]
    print("defined cfl")

    init_palm(coords_for_line,image_width,image_height)
    print("init")

    # process values
    intersection_coords = calculate_intersection_coordinate(image,intersection_coords,coords_for_line)
    print("calculated")


    ret,thresh = utils.threshold(image)
    print("threshold")

    mask = np.zeros(thresh.shape).astype(thresh.dtype)
    print("mask")

    cv2.fillPoly(mask, [intersection_coords], [255, 255, 255])
    print("fillpoly")

    print("done!")
    return cv2.bitwise_and(image,image,mask = mask)







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
