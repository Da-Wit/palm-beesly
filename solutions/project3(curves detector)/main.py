import cv2
import copy
from lines import Lines
import numpy as np
import timeit


# 인풋으로 받은 캐니처리만 된 이미지에서 좌, 우, 위, 아래로 grayscle값이
# min_grayscle 매개변수보다 큰 부분부터 잘라낸 이미지를 리턴함
# 즉, 쓸데없는 빈 공간을 제거해서 연산을 줄임
# TODO raw input image 가지고 ROI를 구하도록 바꾸기
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
    height, width = img_param.shape[:2]

    if is_horizontal:
        first_for_loop_max = width
        second_for_loop_max = height
    else:
        first_for_loop_max = height
        second_for_loop_max = width

    lines = Lines()
    for i in range(first_for_loop_max):
        for j in range(second_for_loop_max):
            if is_horizontal:
                x = i
                y = j
            else:
                x = j
                y = i

            if img_param[y][x] > min_grayscale < max_grayscale:
                lines.handle_point([x, y], max_line_distance)

        if is_horizontal:
            x = i
            y = 0
        else:
            x = 0
            y = i
        lines.renew_work_area([x, y], max_line_distance, is_horizontal)

    return lines


# 이미지와 값 조정 변수를 넣어주면 최종적으로 시각화된 이미지를 가로, 세로로 나눠 리턴함
# 외부에서 최종적으로 사용할 함수
def main(img_param, min_grayscale, max_grayscale, min_line_length, max_line_distance,
         number_of_lines_to_leave, flattening_distance, both=True, is_horizontal=True):
    copied = copy.deepcopy(img_param)
    height, width = copied.shape[:2]
    horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)
    vertical_img = np.zeros((height, width, 1), dtype=np.uint8)

    # Finding lines part
    start = timeit.default_timer()
    horizontal_lines = find_one_orientation_lines(copied, min_grayscale, max_grayscale, max_line_distance,
                                                  is_horizontal=True)
    vertical_lines = find_one_orientation_lines(copied, min_grayscale, max_grayscale, max_line_distance,
                                                is_horizontal=True)
    stop = timeit.default_timer()
    print(round(stop - start, 6))

    # Filtering part
    # horizontal_lines.filter_by_line_length(min_line_length)
    # vertical_lines.filter_by_line_length(min_line_length)

    # Leaving long lines part
    horizontal_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
    vertical_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)

    # cv2.imshow("hori_before_flattening", hori_before_flattening)
    # cv2.imshow("vert_before_flattening", vert_before_flattening)

    # Flattening part
    # horizontal_lines.flatten(flattening_distance, is_horizontal=True)
    # vertical_lines.flatten(flattening_distance, is_horizontal=True)

    # Visualizing part
    horizontal_img = horizontal_lines.visualize_lines(horizontal_img, color=True)
    vertical_img = horizontal_lines.visualize_lines(vertical_img, color=True)

    cv2.imshow("hori before thinning", horizontal_img)
    cv2.imshow("vert before thinning", vertical_img)

    hori_gray = cv2.cvtColor(horizontal_img, cv2.COLOR_BGR2GRAY)
    _, hori_thresh = cv2.threshold(hori_gray, 1, 255, cv2.THRESH_BINARY)
    thinned_hori = cv2.ximgproc.thinning(
        hori_thresh)  # to use this function paste it: pip install opencv-contrib-python

    vert_gray = cv2.cvtColor(vertical_img, cv2.COLOR_BGR2GRAY)
    _, vert_thresh = cv2.threshold(vert_gray, 1, 255, cv2.THRESH_BINARY)
    thinned_vert = cv2.ximgproc.thinning(vert_thresh)

    if both:
        return thinned_hori, thinned_vert
    elif is_horizontal:
        return thinned_hori
    else:
        return thinned_vert


if __name__ == "__main__":
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

    hori, vert = main(img,
                      min_grayscale,
                      max_grayscale,
                      min_line_length,
                      max_line_distance,
                      number_of_lines_to_leave,
                      flattening_distance)

    cv2.imshow("original", img)

    cv2.imshow("hori", hori)
    cv2.imshow("vert", vert)

    print("Done")

    cv2.waitKey(0)
