import cv2
import numpy as np
import os
import mediapipe as mp

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
    Print(maxcnt)
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
            print("results:",results.multi_handedness)

            if not results.multi_hand_landmarks:
              return None
            return results.multi_hand_landmarks[0].landmark
