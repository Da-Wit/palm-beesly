import cv2


def on_min_gray_changed(trackbar_val, window_name, callback):
    min_gray = trackbar_val
    max_gray = cv2.getTrackbarPos('max_gray', window_name)
    line_distance = cv2.getTrackbarPos('line_distance', window_name)
    flattening = cv2.getTrackbarPos('flattening', window_name)

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv2.imshow(window_name, rst)


def on_max_gray_changed(trackbar_val, window_name, callback):
    min_gray = cv2.getTrackbarPos('min_gray', window_name)
    max_gray = trackbar_val
    line_distance = cv2.getTrackbarPos('line_distance', window_name)
    flattening = cv2.getTrackbarPos('flattening', window_name)

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv2.imshow(window_name, rst)


def on_line_distance_changed(trackbar_val, window_name, callback):
    min_gray = cv2.getTrackbarPos('min_gray', window_name)
    max_gray = cv2.getTrackbarPos('max_gray', window_name)
    line_distance = trackbar_val
    flattening = cv2.getTrackbarPos('flattening', window_name)

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv2.imshow(window_name, rst)


def on_flattening_distance_changed(trackbar_val, window_name, callback):
    min_gray = cv2.getTrackbarPos('min_gray', window_name)
    max_gray = cv2.getTrackbarPos('max_gray', window_name)
    line_distance = cv2.getTrackbarPos('line_distance', window_name)
    flattening = trackbar_val

    rst = callback(min_gray,
                   max_gray,
                   line_distance,
                   flattening)

    cv2.imshow(window_name, rst)
