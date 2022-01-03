# This file focus on horizontal_hline_list and vertical_hline_list, not nline3

import cv2
from hline import *
import copy
import solutions.utils as utils
import trackbar


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
    for_debugging = copy.deepcopy(img_param) * 0
    # for_debugging = cv2.cvtColor(for_debugging, cv2.COLOR_GRAY2RGB)
    if is_horizontal:
        first_for_loop_max = w
        second_for_loop_max = h
    else:
        first_for_loop_max = h
        second_for_loop_max = w

    result = []
    min_j_gap = 3
    for i in range(0, first_for_loop_max):
        find = False
        prev_j = 0
        for j in range(0, second_for_loop_max):
            if is_horizontal:
                x = i
                y = j
            else:
                x = j
                y = i
            if img_param[y][x] > min_grayscale and find is False:
                # print("Black2White")
                find = True
                append_point(result, x, y, max_line_distance)
                prev_j = j

            elif img_param[y][x] <= min_grayscale and find is True:
                # print("White2Black")
                find = False
                if j - prev_j > min_j_gap:
                    append_point(result, x, y, max_line_distance)

            # temp = copy.deepcopy(for_debugging)
            # cv2.circle(temp, (x, y), max_line_distance, (255, 0, 0), 1)
            # temp[y][x] = [0, 0, 255]
            # cv2.imshow("img_on_progress", utils.resize(temp, height=1000))

            # k = cv2.waitKey(0)
            # if k == 27:  # Esc key to stop
            #     cv2.destroyAllWindows()
            #     exit(0)

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


def visualize_hline(hline_list, img_param, imshow=False, color=False):
    copied_img = copy.deepcopy(img_param)
    if color:
        copied_img = cv2.cvtColor(copied_img, cv2.COLOR_GRAY2BGR)

    for hline in hline_list:
        for p in hline.pointlist:
            if color:
                copied_img[p[1]][p[0]] = hline.color
            else:
                copied_img[p[1]][p[0]] = 255

        if imshow:
            cv2.imshow("img_on_progress", utils.resize(copied_img, width=600))

            k = cv2.waitKey(0)
            if k == 27:  # Esc key to stop
                cv2.destroyAllWindows()
                exit(0)
    return copied_img


def get_hlines(img, min_grayscale, min_line_length, max_line_distance=3):
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

    horizontal_hline_list, vertical_hline_list = get_hlines(
        img, min_grayscale, min_line_length, max_line_distance)

    horizontal_img = visualize_hline(
        horizontal_hline_list, horizontal_img, imshow=False, color=True)
    vertical_img = visualize_hline(
        vertical_hline_list, vertical_img, imshow=False, color=True)

    return horizontal_img, vertical_img


image_path = "C:/Users/USER/workspace/palm/test_img/edit2.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image is empty!!")
    exit(0)

img = get_roi(img)

# Default value is 0
# min_grayscale = 63
min_grayscale = 70

# The minimum number of dots in one line
# Default value is 4
min_line_length = 6

# Default value is 3
max_line_distance = 6

img2, img3 = get_calculated_img(img, min_grayscale, min_line_length, max_line_distance)

result = img2 + img3

# # window_name을 이름으로 하는 윈도우를 만들어 놓음으로써 해당 윈도우에 트랙바를 달 수 있게 함
window_name = 'result'

cv2.imshow("original", utils.resize(img, width=600))
cv2.imshow("vertical", utils.resize(img3, width=600))
cv2.imshow("horizontal", utils.resize(img2, width=600))
cv2.imshow(window_name, utils.resize(result, width=600))
cv2.waitKey(0)
