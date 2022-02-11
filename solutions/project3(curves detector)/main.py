import cv2
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
    result = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 0)
    return result


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
    image_path = "/Users/david/workspace/palm-beesly/test_img/sample1.4.png"
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Image is empty!!")
        exit(1)
    return img


def get_thinned(bgr_img):
    gray = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    result = cv.ximgproc.thinning(thresh)  # to use this function paste it: pip install opencv-contrib-python
    return result


if __name__ == "__main__":
    grayed = init_imgs()
    img, vertices = get_roi(grayed)

    thr = adaptive_threshold(img)
    cv.imshow("adaptiveThreshold", thr)

    max_line_distance = 5  # Default value is 3
    min_line_length = 4  # The minimum number of dots in one line, Default value is 4
    number_of_lines_to_leave = 10  # Default value is 10

    height, width = thr.shape[:2]
    horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)

    start = timeit.default_timer()
    lines = find_one_orientation_lines(thr, max_line_distance, is_horizontal=True)
    lines.imshow("just found", height, width)
    stop = timeit.default_timer()
    print("Finding part", round(stop - start, 6))

    start = timeit.default_timer()
    lines.combine(max_distance=1)
    lines.imshow("combined", height, width)
    stop = timeit.default_timer()
    print("combining part", round(stop - start, 6))

    start = timeit.default_timer()
    lines.filter_by_line_length(min_line_length)
    lines.leave_long_lines(number_of_lines_to_leave)
    filtered = lines.visualize_lines(height, width)
    cv.imshow("filtered", filtered)
    stop = timeit.default_timer()
    print("Filtering part", round(stop - start, 6))

    start = timeit.default_timer()
    thinned = get_thinned(filtered)
    cv.imshow("thinned", thinned)
    stop = timeit.default_timer()
    print("Thinning part", round(stop - start, 6))

    cv.imshow("original", img)
    print("Done")

    cv.waitKey(0)
