# This file focus on horizontal_hline_list and vertical_hline_list, not nline3

import copy
import cv2
from hline import *
import utils


# h, w = img.shape[:2]
# horizontal_hline_list = []  # 가로 선 나올때마다 추가됨
# vertical_hline_list = []  # 세로 선 나올때마다 추가됨

# 맨 처음 for 문 도는데 처음이 선임
# 맨 처음 for 문 도는데 처음이 선이 아님
# 선 인식 됐다가 인식 안된 상황
# 선 인식 안 됐다가 인식 된 상황
# 1번째 for문 바뀌는 상황

def get_roi(img, min_grayscale=0):
    copied = copy.deepcopy(img)
    height, width = copied.shape[:2]

    topmost = 0
    downmost = height - 1
    leftmost = 0
    rightmost = width - 1

    done = False

    for x in range(width):
        if done is True:
            break
        for y in range(height):
            if copied[y][x] > min_grayscale:
                leftmost = x
                done = True

    done = False

    for y in range(height):
        if done is True:
            break
        for x in range(width):
            if copied[y][x] > min_grayscale:
                topmost = y
                done = True

    done = False

    for y in range(height - 1, -1, -1):
        if done is True:
            break
        for x in range(width):
            if copied[y][x] > min_grayscale:
                downmost = y
                done = True

    done = False

    for x in range(width - 1, -1, -1):
        if done is True:
            break
        for y in range(height):
            if copied[y][x] > min_grayscale:
                rightmost = x
                done = True

    return copied[topmost:downmost, leftmost:rightmost]


def find_one_orientation_lines(img_param, min_grayscale, max_line_distance, is_horizontal):
    h, w = img_param.shape[:2]
    if is_horizontal:
        first_for_loop_max = w
        second_for_loop_max = h
    else:
        first_for_loop_max = h
        second_for_loop_max = w

    result = []
    for i in range(0, first_for_loop_max):
        find = False
        for j in range(0, second_for_loop_max):
            if is_horizontal:
                x = i
                y = j
            else:
                x = j
                y = i
            if img_param[y][x] > min_grayscale and find is False:
                find = True
                set_line_info(result, x, y, max_line_distance)
            if img_param[y][x] <= min_grayscale and find is True:
                find = False
    return result


# 가로 선 찾기
def find_horizontal_lines(img_param, min_grayscale, max_line_distance):
    return find_one_orientation_lines(img_param, min_grayscale, max_line_distance,
                                      is_horizontal=True)


# 세로 선 찾기
def find_vertical_lines(img_param, min_grayscale, max_line_distance):
    return find_one_orientation_lines(img_param, min_grayscale, max_line_distance,
                                      is_horizontal=False)


def filter_hline_by_line_length(hline_list, min_length):
    copied_list = copy.deepcopy(hline_list)
    for hline in copied_list[:]:
        if len(hline.pointlist) < min_length:
            copied_list.remove(hline)  # 길이가 3픽셀도 안되는 선은 세로선이거나 잡음이므로 지움.
    return copied_list


def visualize_hline(hline_list, img_param, imshow=False):
    copied_img = copy.deepcopy(img_param)
    for hline in hline_list:
        for p in hline.pointlist:
            copied_img[p[1]][p[0]] = 255

        if imshow:
            cv2.imshow("img_on_progress", utils.resize(copied_img, width=400))

            k = cv2.waitKey(0)
            if k == 27:  # Esc key to stop
                cv2.destroyAllWindows()
                exit(0)
    return copied_img


def get_hline(img, min_grayscale, min_line_length, max_line_distance=3):
    horizontal_hline_list = find_horizontal_lines(
        img, min_grayscale, max_line_distance)
    vertical_hline_list = find_vertical_lines(
        img, min_grayscale, max_line_distance)

    horizontal_hline_list = filter_hline_by_line_length(
        horizontal_hline_list, min_line_length)
    vertical_hline_list = filter_hline_by_line_length(
        vertical_hline_list, min_line_length)

    return horizontal_hline_list, vertical_hline_list


def get_calculated_img(img, min_grayscale, min_line_length, max_line_distance=3):
    horizontal_img = copy.deepcopy(img) * 0
    vertical_img = copy.deepcopy(img) * 0

    horizontal_hline_list, vertical_hline_list = get_hline(img, min_grayscale, min_line_length, max_line_distance)

    horizontal_img = visualize_hline(horizontal_hline_list, horizontal_img, imshow=False)
    vertical_img = visualize_hline(vertical_hline_list, vertical_img, imshow=False)

    return horizontal_img, vertical_img


image_path = "C:/Users/USER/workspace/palm/test_img/edit5.png"
# image_path = "C:/Users/USER/Desktop/abcd.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = get_roi(img)

# Default value is 0
min_grayscale = 63

# The minimum number of dots in one line
# Default value is 4
min_line_length = 6

# Default value is 3
max_line_distance = 3

img2, img3 = get_calculated_img(img, min_grayscale, min_line_length)

result = img2 + img3

window_name = 'result'

grayscale_trackbar_name = 'min_grayscale'
grayscale_trackbar_start_pos = min_grayscale
max_grayscale_trackbar_val = 255

line_length_trackbar_name = 'line_length'
line_length_trackbar_start_pos = min_line_length
max_line_length_trackbar_val = 15

max_line_distance_trackbar_name = 'max_line_distance'
max_line_distance_trackbar_start_pos = max_line_distance
max_max_line_distance_trackbar_val = 14
# window_name을 이름으로 하는 윈도우를 만들어 놓음으로써 해당 윈도우에 트랙바를 달 수 있게 함
cv2.namedWindow(window_name)


def on_grayscale_trackbar_changed(trackbar_val, img_param):
    min_line_length = cv2.getTrackbarPos(
        line_length_trackbar_name, window_name)

    max_line_distance = cv2.getTrackbarPos(
        max_line_distance_trackbar_name, window_name)

    horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=trackbar_val,
                                                      min_line_length=min_line_length,
                                                      max_line_distance=max_line_distance)

    result_img = horizontal_img + vertical_img

    cv2.imshow(window_name, utils.resize(result_img, width=400))


def on_line_length_trackbar_changed(trackbar_val, img_param):
    min_grayscale = cv2.getTrackbarPos(
        grayscale_trackbar_name, window_name)

    max_line_distance = cv2.getTrackbarPos(
        max_line_distance_trackbar_name, window_name)

    horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=min_grayscale,
                                                      min_line_length=trackbar_val, max_line_distance=max_line_distance)

    result_img = horizontal_img + vertical_img

    cv2.imshow(window_name, utils.resize(result_img, width=400))


def on_max_line_distance_trackbar_changed(trackbar_val, img_param):
    min_grayscale = cv2.getTrackbarPos(
        grayscale_trackbar_name, window_name)

    min_line_length = cv2.getTrackbarPos(
        line_length_trackbar_name, window_name)

    horizontal_img, vertical_img = get_calculated_img(img_param, min_grayscale=min_grayscale,
                                                      min_line_length=min_line_length, max_line_distance=trackbar_val)
    result_img = horizontal_img + vertical_img

    cv2.imshow(window_name, utils.resize(result_img, width=400))


# # 이름이 window_name인 창에 scharr 커널에 곱할 값을 설정하는 트랙바를 만든다
cv2.createTrackbar(grayscale_trackbar_name,
                   window_name,
                   grayscale_trackbar_start_pos,
                   max_grayscale_trackbar_val,
                   lambda val: on_grayscale_trackbar_changed(val, img),
                   )

cv2.createTrackbar(line_length_trackbar_name,
                   window_name,
                   line_length_trackbar_start_pos,
                   max_line_length_trackbar_val,
                   lambda val: on_line_length_trackbar_changed(val, img),
                   )

cv2.createTrackbar(max_line_distance_trackbar_name,
                   window_name,
                   max_line_distance_trackbar_start_pos,
                   max_max_line_distance_trackbar_val,
                   lambda val: on_max_line_distance_trackbar_changed(val, img),
                   )

cv2.imshow("original", utils.resize(img, width=400))
cv2.imshow("vertical", utils.resize(img3, width=400))
cv2.imshow("horizontal", utils.resize(img2, width=400))
cv2.imshow("result", utils.resize(result, width=400))

cv2.waitKey(0)

"""
진행 상황

1단계
 가로로 세로운 선 찾는 정도.

2단계
 세로로도 새로운선찾기

3단계
 찾다가 두께가 너무 두꺼우면 해당 선 취소하기

4단계
 가로선 세로선 가지고 연결점 찾기
 가로선 찾다가 두꺼운점


기울기 계산하기

기울기로 가로선인지 세로선인지 판단해서
 가로선으로 계산할지 세로선으로 계산할지 판단


 어쩌면 가로세로선 구분하지말고
 시작만 가로 세로선에서 하고
 시작하고 나면 가로로 진행할지 세로로 진행할지를 기울기 진행도 가지고
  판단하기??


-가로 선을 찾을때 세로로

"""
