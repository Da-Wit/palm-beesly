import cv2
import copy
import solutions.utils as utils
from lines import Lines
import numpy as np
import timeit
from solutions.zoom.main import zoom

start = timeit.default_timer()


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
            if img_param[y][x] > min_grayscale < max_grayscale and find is False:
                find = True
                lines.handle_point([x, y], max_line_distance)
                prev_j = j

            elif (min_grayscale >= img_param[y][x] or img_param[y][x] >= max_grayscale) and find is True:
                find = False
                if j - prev_j > min_j_gap:
                    lines.handle_point(
                        [x, y], max_line_distance)

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
def main(img_param, min_grayscale, max_grayscale, min_line_length, max_line_distance=3, number_of_lines_to_leave=10):
    cropped = get_roi(img_param)
    cv2.imshow("original",cropped)
    height, width = cropped.shape[:2]

    horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)
    vertical_img = np.zeros((height, width, 1), dtype=np.uint8)

    horizontal_lines = find_horizontal_lines(cropped, min_grayscale, max_grayscale, max_line_distance)
    vertical_lines = find_vertical_lines(cropped, min_grayscale, max_grayscale, max_line_distance)

    # horizontal_lines.filter_by_line_length(min_line_length)
    # vertical_lines.filter_by_line_length(min_line_length)

    horizontal_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
    vertical_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)

    horizontal_img = horizontal_lines.visualize_lines(horizontal_img, color=True)
    vertical_img = vertical_lines.visualize_lines(vertical_img, color=True)

    return horizontal_img, vertical_img


image_path = "C:/Users/think/workspace/palm-beesly/test_img/sample5.4.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image is empty!!")
    exit(1)


img = zoom(img, 0.5)

# Default value is 0
# min_grayscale = 63
min_grayscale = 70

max_grayscale = 200

# The minimum number of dots in one line
# Default value is 4
min_line_length = 10

# Default value is 3
max_line_distance = 5

number_of_lines_to_leave = 10

img2, img3 = main(img, min_grayscale, max_grayscale,
                  min_line_length, max_line_distance, number_of_lines_to_leave)

# result = img2 + img3

stop = timeit.default_timer()

print('Time: ', stop - start)

# cv2.imshow("original", utils.resize(img, width=600))
cv2.imshow("vertical", img3)
cv2.imshow("horizontal", img2)
# cv2.imshow("vertical", utils.resize(img3, width=600))
# cv2.imshow("horizontal", utils.resize(img2, width=600))
# cv2.imshow("result", result)

cv2.waitKey(0)
