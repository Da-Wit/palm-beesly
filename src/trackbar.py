# 이 파일은 cv.createTrackbar의 5번째 인자인 onChange에 사용될 함수들만을 저장하는 파일이다.
# 현재 이 파일은 어디에서도 import되지 않고 있다.

import cv2 as cv


def on_min_gray_changed(trackbar_val, window_name, callback):
    min_gray = trackbar_val

    try:
        max_gray = cv.getTrackbarPos('max_gray', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        max_gray = 200

    try:
        line_distance = cv.getTrackbarPos('line_distance', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        line_distance = 5

    try:
        flattening = cv.getTrackbarPos('flattening', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        flattening = 4

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv.imshow(window_name, rst)


def on_max_gray_changed(trackbar_val, window_name, callback):
    try:
        min_gray = cv.getTrackbarPos('min_gray', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        min_gray = 70

    max_gray = trackbar_val

    try:
        line_distance = cv.getTrackbarPos('line_distance', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        line_distance = 5

    try:
        flattening = cv.getTrackbarPos('flattening', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        flattening = 4

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv.imshow(window_name, rst)


def on_line_distance_changed(trackbar_val, window_name, callback):
    try:
        min_gray = cv.getTrackbarPos('min_gray', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        min_gray = 70

    try:
        max_gray = cv.getTrackbarPos('max_gray', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        max_gray = 200

    line_distance = trackbar_val

    try:
        flattening = cv.getTrackbarPos('flattening', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        flattening = 4

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv.imshow(window_name, rst)


def on_flattening_distance_changed(trackbar_val, window_name, callback):
    try:
        min_gray = cv.getTrackbarPos('min_gray', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        min_gray = 70

    try:
        max_gray = cv.getTrackbarPos('max_gray', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        max_gray = 200

    try:
        line_distance = cv.getTrackbarPos('line_distance', window_name)
    except cv.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        line_distance = 5

    flattening = trackbar_val

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv.imshow(window_name, rst)
