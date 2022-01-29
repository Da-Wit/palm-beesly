import cv2


def on_min_gray_changed(trackbar_val, window_name, callback):
    min_gray = trackbar_val

    try:
        max_gray = cv2.getTrackbarPos('max_gray', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        max_gray = 200

    try:
        line_distance = cv2.getTrackbarPos('line_distance', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        line_distance = 5

    try:
        flattening = cv2.getTrackbarPos('flattening', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        flattening = 4

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv2.imshow(window_name, rst)


def on_max_gray_changed(trackbar_val, window_name, callback):
    try:
        min_gray = cv2.getTrackbarPos('min_gray', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        min_gray = 70

    max_gray = trackbar_val

    try:
        line_distance = cv2.getTrackbarPos('line_distance', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        line_distance = 5

    try:
        flattening = cv2.getTrackbarPos('flattening', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        flattening = 4

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv2.imshow(window_name, rst)


def on_line_distance_changed(trackbar_val, window_name, callback):
    try:
        min_gray = cv2.getTrackbarPos('min_gray', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        min_gray = 70

    try:
        max_gray = cv2.getTrackbarPos('max_gray', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        max_gray = 200

    line_distance = trackbar_val

    try:
        flattening = cv2.getTrackbarPos('flattening', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        flattening = 4

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv2.imshow(window_name, rst)


def on_flattening_distance_changed(trackbar_val, window_name, callback):
    try:
        min_gray = cv2.getTrackbarPos('min_gray', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        min_gray = 70

    try:
        max_gray = cv2.getTrackbarPos('max_gray', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        max_gray = 200

    try:
        line_distance = cv2.getTrackbarPos('line_distance', window_name)
    except cv2.error:
        print("I don't know why this error caused. I'll just pass the default value.")
        line_distance = 5

    flattening = trackbar_val

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv2.imshow(window_name, rst)
