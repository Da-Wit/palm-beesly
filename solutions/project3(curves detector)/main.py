import cv2
import copy
from lines import Lines
import numpy as np
# import trackbar
import timeit


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


def express_img(img_param, min_grayscale, max_grayscale, is_horizontal):
    height, width = img_param.shape[:2]
    result = []
    if is_horizontal:
        loop_max = width
    else:
        loop_max = height

    for i in range(loop_max):
        if is_horizontal:
            x = i
            temp = [[img_param[y][x], x, y] for y in range(height) if
                    img_param[y][x] > min_grayscale < max_grayscale]
        else:
            y = i
            temp = [[img_param[y][x], x, y] for x in range(width) if
                    img_param[y][x] > min_grayscale < max_grayscale]

        if len(temp) > 0:
            result += temp
    return np.array(result, dtype=int)


# 한 방향으로의 hline들의 리스트, nline을 리턴함
def find_one_orientation_lines(expressed, max_line_distance, is_horizontal):
    lines = Lines()
    min_j_gap = 3

    if is_horizontal:
        i = 1  # x
        j = 2  # y
    else:
        i = 2  # y
        j = 1  # x

    pre_i = expressed[0][i]
    pre_j = expressed[0][j]
    continued_i = expressed[0][i]
    continued_j = expressed[0][j]
    was_continued = False

    for idx in range(1, len(expressed)):
        current = expressed[idx]
        _, x, y = current

        if pre_i != current[i]:
            lines.renew_work_area([x, y], max_line_distance, is_horizontal)
            lines.handle_point([x, y], max_line_distance)
            if was_continued:
                if abs(continued_j - pre_j) > min_j_gap:
                    if is_horizontal:
                        lines.handle_point([continued_i, continued_j], max_line_distance)
                    else:
                        lines.handle_point([continued_j, continued_i], max_line_distance)
                was_continued = False
            pre_i = current[i]
            pre_j = current[j]


        elif abs(j - pre_j) > 1:
            lines.handle_point([x, y], max_line_distance)
            if was_continued:
                if abs(continued_j - pre_j) > min_j_gap:
                    if is_horizontal:
                        lines.handle_point([x, continued_j], max_line_distance)
                    else:
                        lines.handle_point([continued_j, y], max_line_distance)
                was_continued = False
            pre_j = current[j]
        else:
            if was_continued is False:
                was_continued = True
                continued_i = current[i]
                continued_j = current[j]

            pre_j = current[j]
            if pre_i != current[i]:
                pre_i = current[i]

    return lines


# 이미지와 값 조정 변수를 넣어주면 최종적으로 시각화된 이미지를 가로, 세로로 나눠 리턴함
# 외부에서 최종적으로 사용할 함수
def main(img_param, min_grayscale, max_grayscale, min_line_length, max_line_distance,
         number_of_lines_to_leave, flattening_distance, both=True, is_horizontal=True):
    print()
    print()
    print()
    sum_timer = 0

    # Copying part
    copied = copy.deepcopy(img_param)
    height, width = copied.shape[:2]

    # expressed = express_img(copied, min_grayscale, max_grayscale)
    hori_img = np.zeros((height, width, 1), dtype=np.uint8)
    vert_img = np.zeros((height, width, 1), dtype=np.uint8)
    expressed = express_img(img_param, min_grayscale, max_grayscale, is_horizontal)
    # Finding lines part
    start = timeit.default_timer()
    horizontal_lines = find_one_orientation_lines(expressed, max_line_distance, is_horizontal=True)
    vertical_lines = find_one_orientation_lines(expressed, max_line_distance, is_horizontal=False)
    stop = timeit.default_timer()
    print("Finding part", stop - start)
    sum_timer += stop - start

    # # Filtering part
    # horizontal_lines.filter_by_line_length(min_line_length)
    # vertical_lines.filter_by_line_length(min_line_length)

    # Leaving long lines part
    start = timeit.default_timer()
    horizontal_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
    vertical_lines.leave_long_lines(number_of_lines_to_leave=number_of_lines_to_leave)
    stop = timeit.default_timer()
    sum_timer += stop - start

    # # Flattening part
    # start = timeit.default_timer()
    # horizontal_lines.flatten(flattening_distance, is_horizontal=True)
    # vertical_lines.flatten(flattening_distance, is_horizontal=True)
    # stop = timeit.default_timer()
    # print("Flattening part", stop - start)
    # sum_timer += stop - start

    # Visualizing part
    start = timeit.default_timer()
    hori_img = horizontal_lines.visualize_lines(hori_img, color=True)
    vert_img = horizontal_lines.visualize_lines(vert_img, color=True)
    stop = timeit.default_timer()
    sum_timer += stop - start

    _, hori_thresh = cv2.threshold(hori_img, 0, 255, cv2.THRESH_BINARY)
    _, vert_thresh = cv2.threshold(vert_img, 0, 255, cv2.THRESH_BINARY)
    # hori_thresh = cv2.cvtColor(cv2.CV_8UC1)
    # cv2.cvtColor(hori_thresh, cv2.COLOR_BGR2GRAY)
    # cv2.cvtColor(vert_thresh, cv2.COLOR_BGR2GRAY)
    # vert_thresh = cv2.cvtColor(cv2.CV_8UC1)
    hori_img = cv2.ximgproc.thinning(cv2.cvtColor(hori_thresh, cv2.COLOR_BGR2GRAY))
    vert_img = cv2.ximgproc.thinning(cv2.cvtColor(vert_thresh, cv2.COLOR_BGR2GRAY))

    print()
    print("sum_timer:", sum_timer)
    if both:
        return hori_img, vert_img
    elif is_horizontal:
        return hori_img
    else:
        return vert_img


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

    # simplified_get_horizontal = lambda min_grayscale, \
    #                                    max_grayscale, \
    #                                    max_line_distance, \
    #                                    flattening_distance: main(img,
    #                                                              min_grayscale,
    #                                                              max_grayscale,
    #                                                              min_line_length,
    #                                                              max_line_distance,
    #                                                              number_of_lines_to_leave,
    #                                                              flattening_distance,
    #                                                              both=False,
    #                                                              is_horizontal=True)
    #
    # simplified_get_vertical = lambda min_grayscale, \
    #                                  max_grayscale, \
    #                                  max_line_distance, \
    #                                  flattening_distance: main(img, min_grayscale,
    #                                                            max_grayscale,
    #                                                            min_line_length,
    #                                                            max_line_distance,
    #                                                            number_of_lines_to_leave,
    #                                                            flattening_distance,
    #                                                            both=False, is_horizontal=False)

    cv2.imshow("original", img)
    cv2.imshow("horizontal", hori)
    cv2.imshow("vertical", vert)

    # on_min_gray_changed_hori = lambda track_val: trackbar.on_min_gray_changed(track_val,
    #                                                                           'horizontal',
    #                                                                           simplified_get_horizontal)
    # on_max_gray_changed_hori = lambda track_val: trackbar.on_max_gray_changed(track_val,
    #                                                                           'horizontal',
    #                                                                           simplified_get_horizontal)
    # on_line_distance_changed_hori = lambda track_val: trackbar.on_line_distance_changed(track_val,
    #                                                                                     'horizontal',
    #                                                                                     simplified_get_horizontal)
    # on_flattening_distance_changed_hori = lambda track_val: trackbar.on_flattening_distance_changed(track_val,
    #                                                                                                 'horizontal',
    #                                                                                                 simplified_get_horizontal)

    # cv2.createTrackbar('min_gray', 'horizontal', min_grayscale, 255, on_min_gray_changed_hori)
    # cv2.createTrackbar('max_gray', 'horizontal', max_grayscale, 255, on_max_gray_changed_hori)
    # cv2.createTrackbar('line_distance', 'horizontal', max_line_distance, 30, on_line_distance_changed_hori)
    # cv2.createTrackbar('flattening', 'horizontal', flattening_distance, 15, on_flattening_distance_changed_hori)

    print("Done")

    cv2.waitKey(0)
