import cv2 as cv
import copy as cp
from lines import Lines
import timeit
import numpy as np
import utils


def get_roi(img_param, min_grayscale=0):
    copied = img_param.copy()
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


# Now it's same with find_one_orientation_lines
# Please update this function too, when updating function find_one_orientation_lines
def find_one_orientation_lines_with_debugging(img_param, max_line_distance, is_horizontal):
    height, width = img_param.shape[:2]
    img_for_debugging = np.zeros((height, width, 1), dtype=np.uint8)
    img_for_debugging = cv.cvtColor(img_for_debugging, cv.COLOR_GRAY2BGR)

    if is_horizontal:
        first_for_loop_max = width
        second_for_loop_max = height
    else:
        first_for_loop_max = height
        second_for_loop_max = width

    lines = Lines()
    unique_num = 0
    skipped = False
    num_of_skip = 40

    for i in range(first_for_loop_max):
        if is_horizontal:
            x = i
        else:
            y = i

        for j in range(second_for_loop_max):
            if skipped:
                num_of_skip -= 1

            if is_horizontal:
                y = j
            else:
                x = j

            if skipped:
                if img_param[y][x] == 255:
                    lines.handle_point([x, y], max_line_distance, unique_num, is_horizontal, debug=True,
                                       img_for_debugging=img_for_debugging)
                    unique_num += 1
                if num_of_skip == 0:
                    skipped = False
                    img_for_debugging *= 0
                    for lineOne in lines.line_list:
                        for point in lineOne.all_point_list:
                            x2, y2 = point
                            img_for_debugging[y2][x2] = lineOne.color
            else:
                if img_param[y][x] == 255:
                    lines.handle_point([x, y], max_line_distance, unique_num, is_horizontal, debug=True,
                                       img_for_debugging=img_for_debugging)
                    unique_num += 1
                prev_bgr = cp.deepcopy(img_for_debugging[y][x])
                img_for_debugging[y][x] = [0, 0, 255]
                cv.imshow("debugging", img_for_debugging)
                k = cv.waitKey(0)
                # for문 도중 Esc를 누르면 프로그램이 종료되게 함
                if k == 113:  # q key to stop
                    exit(0)
                if k == 115:  # s key to skip
                    skipped = True
                    num_of_skip = 20
                    img_for_debugging[y][x] = prev_bgr
                else:
                    img_for_debugging[y][x] = prev_bgr

        if is_horizontal:
            x = i
            y = 0
        else:
            x = 0
            y = i
        lines.renew_work_area([x, y], max_line_distance, is_horizontal)

    return lines


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
        if is_horizontal:
            x = i
        else:
            y = i
        for j in range(second_for_loop_max):
            if is_horizontal:
                y = j
            else:
                x = j

            if img_param[y][x] == 255:
                lines.handle_point([x, y], max_line_distance, unique_num, is_horizontal)
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
    image_path = "/Users/david/workspace/palm-beesly/test_img/sample7.4.png"
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if image is None:
        print("Image is empty!!")
        exit(1)
    return image


def thin(bgr_img):
    gray = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    result = cv.ximgproc.thinning(thresh)  # to use this function paste it: pip install opencv-contrib-python
    return result


if __name__ == "__main__":
    grayed = init_imgs()
    img, vertices = get_roi(grayed)
    thr = utils.adaptive_threshold(img, box_size=11)

    cv.imshow("original", img)
    cv.imshow("adaptiveThreshold", thr)

    max_line_distance = 1.3  # Default value is 3
    combining_distance = 1
    min_line_length = 100  # The minimum number of dots in one line, Default value is 4
    number_of_lines_to_leave = 6  # Default value is 10

    height, width = thr.shape[:2]

    start = timeit.default_timer()
    # lines = find_one_orientation_lines_with_debugging(thr, max_line_distance, is_horizontal=False)
    lines = find_one_orientation_lines(thr, max_line_distance, is_horizontal=False)
    lines.imshow("just found", height, width)
    stop = timeit.default_timer()
    print("Finding part", round(stop - start, 6))

    start = timeit.default_timer()
    # lines.filter_by_line_length(min_line_length)
    lines.sort()
    del lines.line_list[0]
    lines.leave_long_lines(number_of_lines_to_leave)
    filtered = lines.visualize_lines(height, width)
    cv.imshow("filtered", filtered)
    stop = timeit.default_timer()
    print("Filtering part", round(stop - start, 6))

    # start = timeit.default_timer()
    # thinned = thin(filtered)
    # cv.imshow("thinned", thinned)
    # stop = timeit.default_timer()
    # print("Thinning part", round(stop - start, 6))

    print("Done")

    cv.waitKey(0)
