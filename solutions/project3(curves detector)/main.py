import cv2
import copy
from lines import Lines
import numpy as np
import timeit

timer_sum = 0
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
def main(img_param, min_grayscale, max_grayscale, min_line_length, max_line_distance=3, number_of_lines_to_leave=10,
         timer_sum=0):
    # Getting roi part
    start = timeit.default_timer()
    cropped = get_roi(img_param)
    height, width = cropped.shape[:2]
    stop = timeit.default_timer()
    cv2.imshow("original", cropped)

    print("Getting roi: ", stop - start)
    timer_sum += stop - start

    # Finding lines part
    start = timeit.default_timer()
    horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)
    vertical_img = np.zeros((height, width, 1), dtype=np.uint8)
    horizontal_lines = find_horizontal_lines(cropped, min_grayscale, max_grayscale, max_line_distance)
    vertical_lines = find_vertical_lines(cropped, min_grayscale, max_grayscale, max_line_distance)
    stop = timeit.default_timer()

    print("Finding lines: ", stop - start)
    timer_sum += stop - start

    # Filtering part
    start = timeit.default_timer()
    horizontal_lines.filter_by_line_length(min_line_length)
    vertical_lines.filter_by_line_length(min_line_length)
    stop = timeit.default_timer()
    print("Filtering: ", stop - start)
    timer_sum += stop - start

    # Leaving long lines part
    start = timeit.default_timer()
    horizontal_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
    vertical_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
    stop = timeit.default_timer()
    print("Leaving long lines: ", stop - start)
    timer_sum += stop - start

    temp_max_distance = 1
    horizontal_lines.flatten(temp_max_distance, is_horizontal=True)
    vertical_lines.flatten(temp_max_distance, is_horizontal=False)

    # Visualizing part
    start = timeit.default_timer()
    horizontal_img = horizontal_lines.visualize_lines(horizontal_img, color=True)
    vertical_img = vertical_lines.visualize_lines(vertical_img, color=True)
    stop = timeit.default_timer()
    print("Visualizing: ", stop - start)

    print("Timer sum", timer_sum)
    return horizontal_img, vertical_img


image_path = "C:/Users/think/workspace/palm-beesly/test_img/sample5.4.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image is empty!!")
    exit(1)

min_grayscale = 70  # Default value is 63
max_grayscale = 200

# The minimum number of dots in one line
min_line_length = 10  # Default value is 4
max_line_distance = 5  # Default value is 3
number_of_lines_to_leave = 10  # Default value is 10

stop = timeit.default_timer()
# Initializing part
timer_sum += stop - start  # start lies on top of file.
print("Initializing: ", stop - start)

img2, img3 = main(img, min_grayscale, max_grayscale,
                  min_line_length, max_line_distance, number_of_lines_to_leave, timer_sum)

cv2.imshow("vertical", img3)
cv2.imshow("horizontal", img2)
cv2.waitKey(0)
