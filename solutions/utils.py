import cv2
import numpy as np
import os
import mediapipe as mp
import calculator

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

resultDirectory = 'C:/Users/USER/workspace/palm/results'
imageDirectory = 'C:/Users/USER/workspace/palm/images/sample2.png'

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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img,None,10,7,21)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    return cv2.Canny(img, 40, 80)

def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    return cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)

def threshold(image):
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    return ret,thresh


def get_max_contour(contours):
    max = 0
    maxcnt = None
    for cnt in contours :
        area = cv2.contourArea(cnt)
        if(max < area) :
            max = area
            maxcnt = cnt
    return maxcnt

def remove_bground(image):
    ret,thresh = threshold(image)
    #경계선 찾음
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 영역 찾기

    maxcnt = get_max_contour(contours)
    mask = np.zeros(thresh.shape).astype(thresh.dtype)

    # 경계선 내부 255로
    cv2.fillPoly(mask, [maxcnt], [255, 255, 255])
    return cv2.bitwise_and(image,image,mask = mask)

def get_hand_form(image):
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

# test codes

# 숫자가 크면 클수록 pip와 mcp 사이의 간격이 커집니다.
# 소수도 가능하고, 에러가 나면 이 수를 키우거나 줄이면 됩니다.
EXTREME_CUTTING_RATIO = 6

def get_intersection(image,coord1,coord2):
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

image = cv2.imread("C:/Users/USER/workspace/palm/images/sample2.png")
image = resize(image,height=650)
landmark = get_hand_form(image)
img = image.copy()
# img = cv2.flip(image, 1)
image_height, image_width, _ = img.shape


wrist_coord = (
          int(landmark[mp_hands.HandLandmark.WRIST].x * image_width),
          int(landmark[mp_hands.HandLandmark.WRIST].y * image_height),
            )
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
pip_coords, mcp_coords = init_finger_coords(pip_coords,mcp_coords,image_width,image_height)

length = len(pip_coords)

intersections = np.empty((length+1,2),dtype=np.int32)

intersections[0] = wrist_coord

for i in range(len(pip_coords)):
    intersections[i+1] = get_intersection(img,pip_coords[i],mcp_coords[i])
    # cv2.circle(img,intersections[i+1],1,(255,0,0),2)
# for i in range(len(pip_coords)):
#     cv2.circle(img,mcp_coords[i],1,(255,0,0),2)

# cv2.circle(img,wrist_coord,10,(255,0,255),3)
cv2.imshow("img",img)
cv2.waitKey(0)
