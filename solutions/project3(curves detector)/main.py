import cv2
import copy
import solutions.utils as utils
from lines import Lines
import numpy as np


# 인풋으로 받은 캐니처리만 된 이미지에서 좌, 우, 위, 아래로 grayscle값이
# min_grayscle 매개변수보다 큰 부분부터 잘라낸 이미지를 리턴함
# 즉, 쓸데없는 빈 공간을 제거해서 연산을 줄임
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


# 한 방향으로의 hline들의 리스트, nline을 리턴함
def find_one_orientation_lines(img_param, min_grayscale, max_line_distance, is_horizontal):
    h, w = img_param.shape[:2]
    if is_horizontal:
        first_for_loop_max = w
        second_for_loop_max = h
    else:
        first_for_loop_max = h
        second_for_loop_max = w
    lines = Lines()
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
                find = True
                lines.append_point([x, y], max_line_distance)
                prev_j = j

            elif img_param[y][x] <= min_grayscale and find is True:
                find = False
                if j - prev_j > min_j_gap:
                    lines.append_point([x, y], max_line_distance)

    return lines


# find_one_orientation_lines 함수를 사용해 가로선 nline 찾기
def find_horizontal_lines(img_param, min_grayscale, max_line_distance):
    return find_one_orientation_lines(img_param, min_grayscale, max_line_distance,
                                      is_horizontal=True)


# find_one_orientation_lines 함수를 사용해 세로선 nline 찾기
def find_vertical_lines(img_param, min_grayscale, max_line_distance):
    return find_one_orientation_lines(img_param, min_grayscale, max_line_distance,
                                      is_horizontal=False)


# 선 길이가 일정 길이 이하인 선이 필터링된 nline를 리턴함
def filter_hline_by_line_length(lines, min_length):
    line_list = copy.deepcopy(lines.line_list)
    for hline in line_list:
        if len(hline.point_list) < min_length:
            line_list.remove(hline)  # 길이가 3픽셀도 안되는 선은 세로선이거나 잡음이므로 지움.
    return line_list


# nline을 시각화함
def visualize_hline(hline_list, img_param, imshow=False, color=False):
    copied_img = copy.deepcopy(img_param)
    if color:
        copied_img = cv2.cvtColor(copied_img, cv2.COLOR_GRAY2BGR)

    for hline in hline_list:
        for p in hline.point_list:
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


# 필터링된 가로, 세로 nline을 한 번에 리턴함
def get_hlines(img, min_grayscale, min_line_length, max_line_distance=3):
    horizontal_hline_list = find_horizontal_lines(
        img, min_grayscale, max_line_distance)
    vertical_hline_list = find_vertical_lines(
        img, min_grayscale, max_line_distance)

    horizontal_hline_list = horizontal_hline_list.filter_hline_by_line_length(min_line_length)
    vertical_hline_list = vertical_hline_list.filter_hline_by_line_length(min_line_length)

    return horizontal_hline_list, vertical_hline_list


# 이미지와 값 조정 변수를 넣어주면 최종적으로 시각화된 이미지를 가로, 세로로 나눠 리턴함
# 외부에서 최종적으로 사용할 함수
def get_calculated_img(img_param, min_grayscale, min_line_length, max_line_distance=3):
    # height, width = img_param.shape[:2]
    #
    # horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)
    # vertical_img = np.zeros((height, width, 1), dtype=np.uint8)
    horizontal_img = copy.deepcopy(img) * 0
    vertical_img = copy.deepcopy(img) * 0

    horizontal_hline_list, vertical_hline_list = get_hlines(
        img_param, min_grayscale, min_line_length, max_line_distance)

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

cv2.imshow("original", utils.resize(img, width=600))
cv2.imshow("vertical", utils.resize(img3, width=600))
cv2.imshow("horizontal", utils.resize(img2, width=600))
cv2.imshow("result", utils.resize(result, width=600))
cv2.waitKey(0)
