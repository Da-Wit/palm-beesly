import cv2
from solutions.utils import resize


def on_grayscale_trackbar_changed(window_name, trackbar_val, img_param, line_length_trackbar_name,
                                  max_line_distance_trackbar_name, get_calculated_img):
    min_line_length = cv2.getTrackbarPos(
        line_length_trackbar_name, window_name)

    max_line_distance = cv2.getTrackbarPos(
        max_line_distance_trackbar_name, window_name)

    horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=trackbar_val,
                                                      min_line_length=min_line_length,
                                                      max_line_distance=max_line_distance)

    result_img = horizontal_img + vertical_img

    cv2.imshow(window_name, resize(result_img, width=600))


def on_line_length_trackbar_changed(window_name, trackbar_val, img_param, grayscale_trackbar_name,
                                    max_line_distance_trackbar_name, get_calculated_img):
    min_grayscale = cv2.getTrackbarPos(
        grayscale_trackbar_name, window_name)

    max_line_distance = cv2.getTrackbarPos(
        max_line_distance_trackbar_name, window_name)

    horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=min_grayscale,
                                                      min_line_length=trackbar_val, max_line_distance=max_line_distance)

    result_img = horizontal_img + vertical_img

    cv2.imshow(window_name, resize(result_img, width=600))


def on_max_line_distance_trackbar_changed(window_name, trackbar_val, img_param, grayscale_trackbar_name,
                                          line_length_trackbar_name, get_calculated_img):
    min_grayscale = cv2.getTrackbarPos(
        grayscale_trackbar_name, window_name)

    min_line_length = cv2.getTrackbarPos(
        line_length_trackbar_name, window_name)

    horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=min_grayscale,
                                                      min_line_length=min_line_length, max_line_distance=trackbar_val)
    result_img = horizontal_img + vertical_img

    cv2.imshow(window_name, resize(result_img, width=600))
