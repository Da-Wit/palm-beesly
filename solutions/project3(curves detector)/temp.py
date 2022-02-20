import cv2 as cv
import copy as cp
from lines import Lines


def init_imgs():
    image_path = "/Users/david/workspace/palm-beesly/test_img/sample5.1.png"
    image = cv.imread(image_path)

    if image is None:
        print("Image is empty!!")
        exit(1)
    return image


def adaptive_threshold(img_param):
    result = cv.adaptiveThreshold(img_param, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 1)
    return result


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


def get_thinned(bgr_img):
    gray = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    result = cv.ximgproc.thinning(thresh)  # to use this function paste it: pip install opencv-contrib-python
    return result


# def get_thinned(gray):
#     _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
#     result = cv.ximgproc.thinning(thresh)  # to use this function paste it: pip install opencv-contrib-python
#     return result


if __name__ == "__main__":
    original = init_imgs()

    gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    thr = adaptive_threshold(gray)
    img, vertices = get_roi(thr)

    cv.imshow("original", original)
    cv.imshow("adaptiveThreshold", img)

    max_line_distance = 1.2  # Default value is 3
    combining_distance = 1
    min_line_length = 100  # The minimum number of dots in one line, Default value is 4
    number_of_lines_to_leave = 6  # Default value is 10

    height, width = img.shape[:2]

    # lines = find_one_orientation_lines_with_debugging(thr, max_line_distance, is_horizontal=False)
    lines = find_one_orientation_lines(img, max_line_distance, is_horizontal=False)
    lines.imshow("just found", height, width)

    # lines.filter_by_line_length(min_line_length)
    lines.sort()
    # del lines.line_list[0]
    lines.leave_long_lines(number_of_lines_to_leave)
    filtered = lines.visualize_lines(height, width)
    cv.imshow("filtered", filtered)

    thinned = get_thinned(filtered)
    cv.imshow("thinned", thinned)

    print("Done")

    cv.waitKey(0)
