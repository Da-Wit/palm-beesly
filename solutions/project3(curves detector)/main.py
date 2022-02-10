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

    return copied[topmost:(downmost + 1), leftmost:(rightmost + 1)], {'topmost': topmost, 'downmost': downmost,
                                                                      'leftmost': leftmost, 'rightmost': rightmost}


# 한 방향으로의 hline들의 리스트, nline을 리턴함
def find_one_orientation_lines(img_param, min_grayscale, max_line_distance, is_horizontal):
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

            if min_grayscale < img_param[y][x]:
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


def get_bgr_min_max(original_img, lines, vertices):
    copied = copy.deepcopy(original_img)

    topmost = vertices['topmost']
    downmost = vertices['downmost']
    leftmost = vertices['leftmost']
    rightmost = vertices['rightmost']

    copied = copied[topmost:(downmost + 1), leftmost:(rightmost + 1)]

    b_min, b_max = 256, 0
    g_min, g_max = 256, 0
    r_min, r_max = 256, 0
    for lineOne in lines.line_list:
        for point in lineOne.all_point_list:
            b, g, r = copied[point[1]][point[0]]
            if b < b_min:
                b_min = b
            if g < g_min:
                g_min = g
            if r < r_min:
                r_min = r

            if b > b_max:
                b_max = b
            if g > g_max:
                g_max = g
            if r > r_max:
                r_max = r

    return [[b_min, b_max], [g_min, g_max], [r_min, r_max]]


def init_imgs():
    original_img_path = "/Users/david/workspace/palm-beesly/test_img/sample5.png"
    original_img = cv2.imread(original_img_path)
    image_path = "/Users/david/workspace/palm-beesly/test_img/sample5.4.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original_img is None:
        print("Original image is empty!!")
        exit(1)

    if img is None:
        print("Image is empty!!")
        exit(1)
    return original_img, img


# 이미지와 값 조정 변수를 넣어주면 최종적으로 시각화된 이미지를 가로, 세로로 나눠 리턴함
# 외부에서 최종적으로 사용할 함수
if __name__ == "__main__":
    start = timeit.default_timer()

    original_img, img = init_imgs()
    img, vertices = get_roi(img)

    min_grayscale = 70  # Default value is 63
    max_grayscale = 200
    min_line_length = 4  # The minimum number of dots in one line, Default value is 4
    max_line_distance = 5  # Default value is 3
    number_of_lines_to_leave = 10  # Default value is 10
    flattening_distance = 4  # Default value is 4

    height, width = img.shape[:2]
    horizontal_img = np.zeros((height, width, 1), dtype=np.uint8)

    horizontal_lines = find_one_orientation_lines(img, min_grayscale, max_line_distance, is_horizontal=True)
    rendered_lines = horizontal_lines.visualize_lines(horizontal_img, color=True)
    cv2.imshow("before combining", rendered_lines)

    horizontal_lines.combine(max_distance=4)
    horizontal_lines.filter_by_line_length(min_line_length)
    horizontal_lines.leave_long_lines(number_of_lines_to_leave)
    rendered_filtered = horizontal_lines.visualize_lines(rendered_lines, color=True)
    cv2.imshow("after combining", rendered_filtered)

    grayscale = cv2.cvtColor(rendered_filtered, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    thinned = cv2.ximgproc.thinning(thresh)  # to use this function paste it: pip install opencv-contrib-python

    horizontal_lines = find_one_orientation_lines(thinned, min_grayscale, max_line_distance, is_horizontal=True)
    horizontal_lines.combine(max_distance=4)
    b_minmax, g_minmax, r_minmax = get_bgr_min_max(original_img, horizontal_lines, vertices)
    print(b_minmax, g_minmax, r_minmax)

    b_minmax[0] = round(b_minmax[0] * 0.9)
    b_minmax[1] = round(b_minmax[0] * 1.1)
    g_minmax[0] = round(g_minmax[0] * 0.9)
    g_minmax[1] = round(g_minmax[0] * 1.1)
    r_minmax[0] = round(r_minmax[0] * 0.9)
    r_minmax[1] = round(r_minmax[0] * 1.1)

    cv2.imshow("original", img)
    cv2.imshow("thinned", thinned)

    stop = timeit.default_timer()
    print(round(stop - start, 6))
    print("Done")

    cv2.waitKey(0)
