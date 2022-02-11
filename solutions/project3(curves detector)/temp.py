import cv2 as cv
import copy
from lines import Lines
import timeit
import numpy as np


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

    return copied[topmost:(downmost + 1), leftmost:(rightmost + 1)], {'topmost': topmost, 'downmost': downmost,
                                                                      'leftmost': leftmost, 'rightmost': rightmost}


def adaptive_threshold(img):
    thr = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 0)


# 한 방향으로의 hline들의 리스트, nline을 리턴함
def find_one_orientation_lines(img_param, max_line_distance, is_horizontal):
    height, width = img_param.shape[:2]

    if is_horizontal:
        first_for_loop_max = width
        second_for_loop_max = height
    else:
        first_for_loop_max = height
        second_for_loop_max = width

    lines = Lines()
    unique_num = 0
    for i in range(first_for_loop_max):
        for j in range(second_for_loop_max):
            if is_horizontal:
                x = i
                y = j
            else:
                x = j
                y = i

            if img_param[y][x] == 255:
                lines.handle_point([x, y], max_line_distance, unique_num)
                unique_num += 1

        if is_horizontal:
            x = i
            y = 0
        else:
            x = 0
            y = i
        lines.renew_work_area([x, y], max_line_distance, is_horizontal)

    return lines


def init_imgs():
    image_path = "/Users/david/workspace/palm-beesly/test_img/sample5.4.png"
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Image is empty!!")
        exit(1)
    return img


start = timeit.default_timer()

img_path = "/Users/david/workspace/palm-beesly/test_img/sample5.4.png"
img = cv.imread(img_path)

grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img, vertices = get_roi(grayscale)

thr = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 0)
cv.imshow("adaptiveThreshold", thr)

min_line_length = 4  # The minimum number of dots in one line, Default value is 4
max_line_distance = 5  # Default value is 3
number_of_lines_to_leave = 10  # Default value is 10

height, width = img.shape[:2]
horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)

horizontal_lines = find_one_orientation_lines(thr, max_line_distance, is_horizontal=True)
rendered_lines = horizontal_lines.visualize_lines(thr, color=True)
cv.imshow("just found", rendered_lines)

horizontal_lines.combine(max_distance=1)
combined = horizontal_lines.visualize_lines(rendered_lines, color=True)
cv.imshow("combined", combined)

horizontal_lines.filter_by_line_length(min_line_length)
horizontal_lines.leave_long_lines(number_of_lines_to_leave)
filtered = horizontal_lines.visualize_lines(combined, color=True)
cv.imshow("filtered", filtered)

grayscale = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(grayscale, 1, 255, cv.THRESH_BINARY)
thinned = cv.ximgproc.thinning(thresh)  # to use this function paste it: pip install opencv-contrib-python
cv.imshow("thinned", thinned)

cv.imshow("original", img)

stop = timeit.default_timer()
print(round(stop - start, 6))
print("Done")

cv.waitKey(0)
