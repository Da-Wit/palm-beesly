import numpy as np
from lines import Lines


# import cv2 as cv
# import timeit
#
# img = cv.imread("C:/Users/think/workspace/palm-beesly/test_img/sample5.4.png")
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
#

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
            lines.handle_point([x, y], max_line_distance)
            pre_i = current[i]
            pre_j = current[j]
            if was_continued:
                if abs(continued_j - pre_j) > min_j_gap:
                    if is_horizontal:
                        lines.handle_point([continued_i, continued_j], max_line_distance)
                    else:
                        lines.handle_point([continued_j, continued_i], max_line_distance)
                was_continued = False

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
            was_continued = True
            continued_j = current[j]
            continued_i = current[i]

        if idx % 200 == 199:
            lines.renew_work_area([x, y], max_line_distance)
    return lines

# lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#
# for gray, x, y in lst:
#     print()
#     print()
#     print("gray", gray)
#     print("x", x)
#     print("y", y)
