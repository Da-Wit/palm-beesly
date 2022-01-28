# Legacy
import cv2
from solutions.utils import resize


# window_name = 'result'
#
# grayscale_trackbar_name = 'min_grayscale'
# grayscale_trackbar_start_pos = min_grayscale
# max_grayscale_trackbar_val = 255
#
# line_length_trackbar_name = 'line_length'
# line_length_trackbar_start_pos = min_line_length
# max_line_length_trackbar_val = 15
#
# max_line_distance_trackbar_name = 'max_line_distance'
# max_line_distance_trackbar_start_pos = max_line_distance
# max_max_line_distance_trackbar_val = 14
# # window_name을 이름으로 하는 윈도우를 만들어 놓음으로써 해당 윈도우에 트랙바를 달 수 있게 함
# cv2.namedWindow(window_name)


def on_min_gray_changed(trackbar_val, window_name, callback):
    min_gray = trackbar_val
    max_gray = cv2.getTrackbarPos('max_gray', window_name)
    line_distance = cv2.getTrackbarPos('line_distance', window_name)

    rst = callback(min_gray,
                   max_gray,
                   line_distance)

    cv2.imshow(window_name, rst)


def on_max_gray_changed(trackbar_val, window_name, callback):
    min_gray = cv2.getTrackbarPos('min_gray', window_name)
    max_gray = trackbar_val
    line_distance = cv2.getTrackbarPos('line_distance', window_name)

    rst = callback(min_gray,
                   max_gray,
                   line_distance)

    cv2.imshow(window_name, rst)


def on_line_distance_changed(trackbar_val, window_name, callback):
    min_gray = cv2.getTrackbarPos('min_gray', window_name)
    max_gray = cv2.getTrackbarPos('max_gray', window_name)
    line_distance = trackbar_val

    rst = callback(min_gray,
                   max_gray,
                   line_distance)

    cv2.imshow(window_name, rst)
#
#
# def on_grayscale_trackbar_changed(window_name, trackbar_val, img_param, line_length_trackbar_name,
#                                   max_line_distance_trackbar_name, get_calculated_img):
#     min_line_length = cv2.getTrackbarPos(line_length_trackbar_name, window_name)
#
#     max_line_distance = cv2.getTrackbarPos(max_line_distance_trackbar_name, window_name)
#
#     horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=trackbar_val,
#                                                       min_line_length=min_line_length,
#                                                       max_line_distance=max_line_distance)
#
#     result_img = horizontal_img + vertical_img
#
#     cv2.imshow(window_name, resize(result_img, width=600))
#
#
# def on_line_length_trackbar_changed(window_name, trackbar_val, img_param, grayscale_trackbar_name,
#                                     max_line_distance_trackbar_name, get_calculated_img):
#     min_grayscale = cv2.getTrackbarPos(
#         grayscale_trackbar_name, window_name)
#
#     max_line_distance = cv2.getTrackbarPos(
#         max_line_distance_trackbar_name, window_name)
#
#     horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=min_grayscale,
#                                                       min_line_length=trackbar_val, max_line_distance=max_line_distance)
#
#     result_img = horizontal_img + vertical_img
#
#     cv2.imshow(window_name, resize(result_img, width=600))
#
#
# def on_max_line_distance_trackbar_changed(window_name, trackbar_val, img_param, grayscale_trackbar_name,
#                                           line_length_trackbar_name, get_calculated_img):
#     min_grayscale = cv2.getTrackbarPos(
#         grayscale_trackbar_name, window_name)
#
#     min_line_length = cv2.getTrackbarPos(
#         line_length_trackbar_name, window_name)
#
#     horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=min_grayscale,
#                                                       min_line_length=min_line_length, max_line_distance=trackbar_val)
#     result_img = horizontal_img + vertical_img
#
#     cv2.imshow(window_name, resize(result_img, width=600))
