import cv2
import copy
from lines import Lines
import numpy as np
import trackbar


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
def find_one_orientation_lines(img_param, min_grayscale, max_grayscale, max_line_distance, is_horizontal):
    h, w = img_param.shape[:2]

    if is_horizontal:
        first_for_loop_max = w
        second_for_loop_max = h
    else:
        first_for_loop_max = h
        second_for_loop_max = w

    lines = Lines()
    min_j_gap = 3
    for i in range(first_for_loop_max):
        find = False
        prev_j = 0
        for j in range(second_for_loop_max):
            if is_horizontal:
                x = i
                y = j
            else:
                x = j
                y = i

            if img_param[y][x] > min_grayscale < max_grayscale and find is False:
                find = True
                lines.handle_point([x, y], max_line_distance)
                prev_j = j
            elif (min_grayscale >= img_param[y][x] or img_param[y][x] >= max_grayscale) and find is True:
                find = False
                if j - prev_j > min_j_gap:
                    lines.handle_point([x, y], max_line_distance)
        if is_horizontal:
            x = i
            y = 0
        else:
            x = 0
            y = i
        lines.renew_work_area([x, y], max_line_distance)

    return lines


# find_one_orientation_lines 함수를 사용해 가로선 nline 찾기
def find_horizontal_lines(img_param, min_grayscale, max_grayscale, max_line_distance):
    return find_one_orientation_lines(img_param, min_grayscale, max_grayscale, max_line_distance,
                                      is_horizontal=True)


# find_one_orientation_lines 함수를 사용해 세로선 nline 찾기
def find_vertical_lines(img_param, min_grayscale, max_grayscale, max_line_distance):
    return find_one_orientation_lines(img_param, min_grayscale, max_grayscale, max_line_distance,
                                      is_horizontal=False)


# 이미지와 값 조정 변수를 넣어주면 최종적으로 시각화된 이미지를 가로, 세로로 나눠 리턴함
# 외부에서 최종적으로 사용할 함수

def main(img_param, min_grayscale, max_grayscale, min_line_length, max_line_distance,
         number_of_lines_to_leave, flattening_distance, both=True, is_horizontal=True):
    copied = copy.deepcopy(img_param)
    height, width = copied.shape[:2]

    # Finding lines part
    horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)
    horizontal_lines = find_horizontal_lines(copied, min_grayscale, max_grayscale, max_line_distance)
    vertical_img = np.zeros((height, width, 1), dtype=np.uint8)
    vertical_lines = find_vertical_lines(copied, min_grayscale, max_grayscale, max_line_distance)

    # # Filtering part
    # horizontal_lines.filter_by_line_length(min_line_length)
    # vertical_lines.filter_by_line_length(min_line_length)

    # Leaving long lines part
    horizontal_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
    vertical_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)

    # Visualizing part1
    hori_before_flattening = horizontal_lines.visualize_lines(horizontal_img, color=True)
    vert_before_flattening = vertical_lines.visualize_lines(vertical_img, color=True)

    cv2.imshow("hori_before_flattening", hori_before_flattening)
    cv2.imshow("vert_before_flattening", vert_before_flattening)

    # Flattening part
    horizontal_lines.flatten(flattening_distance, is_horizontal=True)
    vertical_lines.flatten(flattening_distance, is_horizontal=True)

    # Visualizing part2
    horizontal_img = horizontal_lines.visualize_lines(horizontal_img, color=True)
    vertical_img = horizontal_lines.visualize_lines(vertical_img, color=True)

    if both:
        return horizontal_img, vertical_img
    elif is_horizontal:
        return horizontal_img
    else:
        return vertical_img


# def get_both(img_param, min_grayscale, max_grayscale, min_line_length, max_line_distance,
#              number_of_lines_to_leave, flattening_distance):
#     copied = copy.deepcopy(img_param)
#     height, width = copied.shape[:2]
#
#     # Finding lines part
#     horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)
#     horizontal_lines = find_horizontal_lines(copied, min_grayscale, max_grayscale, max_line_distance)
#     vertical_img = np.zeros((height, width, 1), dtype=np.uint8)
#     vertical_lines = find_vertical_lines(copied, min_grayscale, max_grayscale, max_line_distance)
#
#     # # Filtering part
#     # horizontal_lines.filter_by_line_length(min_line_length)
#     # vertical_lines.filter_by_line_length(min_line_length)
#
#     # Leaving long lines part
#     horizontal_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
#     vertical_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
#
#     # Visualizing part1
#     hori_before_flattening = horizontal_lines.visualize_lines(horizontal_img, color=True)
#     vert_before_flattening = vertical_lines.visualize_lines(vertical_img, color=True)
#
#     cv2.imshow("hori_before_flattening", hori_before_flattening)
#     cv2.imshow("vert_before_flattening", vert_before_flattening)
#
#     # Flattening part
#     temp_max_distance = 1
#     horizontal_lines.flatten(temp_max_distance, flattening_distance, is_horizontal=True)
#     vertical_lines.flatten(temp_max_distance, flattening_distance, is_horizontal=True)
#
#     # Visualizing part2
#     horizontal_img = horizontal_lines.visualize_lines(horizontal_img, color=True)
#     vertical_img = horizontal_lines.visualize_lines(vertical_img, color=True)
#
#     return horizontal_img, vertical_img


# def get_horizontal(img_param, min_grayscale, max_grayscale, min_line_length, max_line_distance,
#                    number_of_lines_to_leave, flattening_distance):
#     copied = copy.deepcopy(img_param)
#     height, width = copied.shape[:2]
#
#     # Finding lines part
#     horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)
#     horizontal_lines = find_horizontal_lines(copied, min_grayscale, max_grayscale, max_line_distance)
#
#     # # Filtering part
#     # horizontal_lines.filter_by_line_length(min_line_length)
#
#     # Leaving long lines part
#     horizontal_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
#
#     # Visualizing part1
#     hori_before_flattening = horizontal_lines.visualize_lines(horizontal_img, color=True)
#
#     cv2.imshow("hori_before_flattening", hori_before_flattening)
#
#     # Flattening part
#     temp_max_distance = 2
#     horizontal_lines.flatten(temp_max_distance, is_horizontal=True)
#
#     # Visualizing part2
#     horizontal_img = horizontal_lines.visualize_lines(horizontal_img, color=True)
#
#     return horizontal_img
#
#
# def get_vertical(img_param, min_grayscale, max_grayscale, min_line_length, max_line_distance,
#                  number_of_lines_to_leave, flattening_distance):
#     copied = copy.deepcopy(img_param)
#     height, width = copied.shape[:2]
#
#     # Finding lines part
#     vertical_img = np.zeros((height, width, 1), dtype=np.uint8)
#     vertical_lines = find_vertical_lines(copied, min_grayscale, max_grayscale, max_line_distance)
#
#     # # Filtering part
#     # vertical_lines.filter_by_line_length(min_line_length)
#
#     # Visualizing part1
#     vert_before_flattening = vertical_lines.visualize_lines(horizontal_img, color=True)
#
#     cv2.imshow("vert_before_flattening", vert_before_flattening)
#
#     # Leaving long lines part
#     vertical_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
#
#     # Flattening part2
#     temp_max_distance = 1
#     vertical_lines.flatten(temp_max_distance, is_horizontal=False)
#
#     # Visualizing part
#     vertical_img = vertical_lines.visualize_lines(vertical_img, color=True)
#
#     return vertical_img


image_path = "C:/Users/think/workspace/palm-beesly/test_img/sample5.4.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image is empty!!")
    exit(1)

img = get_roi(img)

min_grayscale = 70  # Default value is 63
max_grayscale = 200

# The minimum number of dots in one line
min_line_length = 10  # Default value is 4
max_line_distance = 5  # Default value is 3
number_of_lines_to_leave = 10  # Default value is 10
flattening_distance = 4  # Default value is 4

horizontal_img, vertical_img = main(img,
                                    min_grayscale,
                                    max_grayscale,
                                    min_line_length,
                                    max_line_distance,
                                    number_of_lines_to_leave,
                                    flattening_distance)

simplified_get_horizontal = lambda min_grayscale, \
                                   max_grayscale, \
                                   max_line_distance, \
                                   flattening_distance: main(img,
                                                             min_grayscale,
                                                             max_grayscale,
                                                             min_line_length,
                                                             max_line_distance,
                                                             number_of_lines_to_leave,
                                                             flattening_distance,
                                                             both=False,
                                                             is_horizontal=True)

simplified_get_vertical = lambda min_grayscale, \
                                 max_grayscale, \
                                 max_line_distance, \
                                 flattening_distance: main(img, min_grayscale,
                                                           max_grayscale,
                                                           min_line_length,
                                                           max_line_distance,
                                                           number_of_lines_to_leave,
                                                           flattening_distance,
                                                           both=False, is_horizontal=False)

cv2.imshow("original", img)
cv2.imshow("horizontal", horizontal_img)
cv2.imshow("vertical", vertical_img)

on_min_gray_changed_hori = lambda track_val: trackbar.on_min_gray_changed(track_val,
                                                                          'horizontal',
                                                                          simplified_get_horizontal)
on_max_gray_changed_hori = lambda track_val: trackbar.on_max_gray_changed(track_val,
                                                                          'horizontal',
                                                                          simplified_get_horizontal)
on_line_distance_changed_hori = lambda track_val: trackbar.on_line_distance_changed(track_val,
                                                                                    'horizontal',
                                                                                    simplified_get_horizontal)
on_flattening_distance_changed_hori = lambda track_val: trackbar.on_flattening_distance_changed(track_val,
                                                                                                'horizontal',
                                                                                                simplified_get_horizontal)

cv2.createTrackbar('min_gray', 'horizontal', min_grayscale, 255, on_min_gray_changed_hori)
cv2.createTrackbar('max_gray', 'horizontal', max_grayscale, 255, on_max_gray_changed_hori)
cv2.createTrackbar('line_distance', 'horizontal', max_line_distance, 30, on_line_distance_changed_hori)
cv2.createTrackbar('flattening', 'horizontal', flattening_distance, 15, on_flattening_distance_changed_hori)

print("Done")

cv2.waitKey(0)
